"""High-variance traffic that extends the base dataset, with a meaning label per sentence.

The base generator (generate_dataset.generate) has a handful of fixed wordings per response type, which favours
exact-match strategies. Real voice-agent traffic comes from an LLM, so it looks more like this:

- Intents: ~40 support intents (orders, refunds, payments, OTP, account, callbacks). Each has several wordings,
  the way an LLM rephrases the same answer. Popularity is Zipf-skewed over intents and over wordings within an intent.
- Near-misses sit next to each other on purpose, because real traffic has them: shipped / delivered / out for
  delivery / delayed / cancelled, today / tomorrow, "on {date}" / "by {date}", "in" / "within" N days,
  refund processed / initiated / rejected, payment successful / failed / pending.
- Values vary like real ones: 6-digit and alphanumeric order IDs (OD…, which no template can slot), amounts written
  ₹ / Rs. / Rs / INR / "rupees", numeric and written dates, 12-hour times.
- Personal sentences (a customer's name) and free-form explanations (product × status × next step) rarely repeat.
- Languages: English, Hindi (including some Hinglish), and a little Kannada. Kannada has no rules file: it runs
  through the language-agnostic defaults, and Kannada suffixes glued to numbers (12/03/2026ರಂದು) cannot be slotted.

Labels: every sentence carries a meaning ID. Two sentences share a label only if playing one in place of the other
would be correct. A name, product or date qualifier ("earlier today", "by" vs "on") is part of the meaning, so the
harness can count semantic hits that played the wrong words.

The Kannada and Hindi wordings were written with AI assistance and should be reviewed by native speakers.
"""

import random

from harness.generate_dataset import (
    CLOSINGS,
    OPENINGS,
    STATIC_BODIES,
    VARIABLE_BODIES,
    Request,
    zipf_choice,
)
from tts_cache.normalize import normalize
from tts_cache.splitter import split_sentences
from tts_cache.strategies.template import find_variables, make_template

LANGUAGE_SHARE = {"en": 0.60, "hi": 0.34, "kn": 0.06}

# (meaning, {language: [wordings]}), most popular first.
INTENTS = [
    (
        "order_shipped",
        {
            "en": [
                "Your order {order} has been shipped.",
                "Good news, your order {order} has been shipped.",
                "Your order {order} is on its way.",
                "We have shipped your order {order}.",
                "Order {order} has left our warehouse and is on its way to you.",
                "Your package for order {order} has been dispatched.",
                "I can confirm that order {order} has been shipped.",
            ],
            "hi": [
                "आपका ऑर्डर {order} भेज दिया गया है।",
                "आपका ऑर्डर {order} रास्ते में है।",
                "हमने आपका ऑर्डर {order} भेज दिया है।",
                "ऑर्डर {order} डिस्पैच हो चुका है।",
                "Aapka order {order} ship ho gaya hai.",
            ],
            "kn": [
                "ನಿಮ್ಮ ಆರ್ಡರ್ {order} ರವಾನೆಯಾಗಿದೆ.",
                "ನಿಮ್ಮ ಆರ್ಡರ್ {order} ದಾರಿಯಲ್ಲಿದೆ.",
                "ಆರ್ಡರ್ {order} ಅನ್ನು ಕಳುಹಿಸಲಾಗಿದೆ.",
            ],
        },
    ),
    (
        "hold_checking",
        {
            "en": [
                "Please hold while I check that for you.",
                "Let me check that for you.",
                "Please wait while I look into that for you.",
                "Give me a moment while I check.",
                "One moment please, let me look that up.",
                "Just a second while I pull up your details.",
            ],
            "hi": [
                "कृपया प्रतीक्षा करें, मैं जाँच कर रहा हूँ।",
                "एक पल रुकिए, मैं देखता हूँ।",
                "Ek minute, main check karta hoon.",
            ],
            "kn": ["ದಯವಿಟ್ಟು ಒಂದು ಕ್ಷಣ ಕಾಯಿರಿ, ನಾನು ಪರಿಶೀಲಿಸುತ್ತೇನೆ."],
        },
    ),
    (
        "refund_processed",
        {
            "en": [
                "Your refund of {amt} has been processed.",
                "We have processed your refund of {amt}.",
                "The refund of {amt} has been processed successfully.",
                "Your {amt} refund is processed.",
            ],
            "hi": [
                "आपका {amt} का रिफंड प्रोसेस हो गया है।",
                "हमने आपका {amt} का रिफंड प्रोसेस कर दिया है।",
                "{amt} का रिफंड सफलतापूर्वक प्रोसेस हो गया है।",
            ],
            "kn": ["ನಿಮ್ಮ {amt} ಮರುಪಾವತಿ ಪ್ರಕ್ರಿಯೆಗೊಂಡಿದೆ."],
        },
    ),
    (
        "order_delivered",
        {
            "en": [
                "Your order {order} has been delivered.",
                "Order {order} was delivered successfully.",
                "Our records show that order {order} has been delivered.",
                "We have delivered your order {order}.",
            ],
            "hi": [
                "आपका ऑर्डर {order} डिलीवर हो गया है।",
                "ऑर्डर {order} की डिलीवरी हो चुकी है।",
                "हमारे रिकॉर्ड के अनुसार ऑर्डर {order} डिलीवर हो चुका है।",
            ],
            "kn": [
                "ನಿಮ್ಮ ಆರ್ಡರ್ {order} ತಲುಪಿಸಲಾಗಿದೆ.",
                "ಆರ್ಡರ್ {order} ಡೆಲಿವರಿ ಆಗಿದೆ.",
            ],
        },
    ),
    (
        "request_received",
        {
            "en": [
                "Your request has been received.",
                "We have received your request.",
                "I have noted your request.",
                "Your request is registered with us.",
            ],
            "hi": [
                "आपका अनुरोध प्राप्त हो गया है।",
                "हमें आपका अनुरोध मिल गया है।",
            ],
            "kn": ["ನಿಮ್ಮ ವಿನಂತಿಯನ್ನು ಸ್ವೀಕರಿಸಲಾಗಿದೆ."],
        },
    ),
    (
        "payment_success",
        {
            "en": [
                "Your payment of {amt} was successful.",
                "We have received your payment of {amt}.",
                "The payment of {amt} went through successfully.",
                "Payment of {amt} is confirmed.",
            ],
            "hi": [
                "आपका {amt} का भुगतान सफल रहा।",
                "हमें आपका {amt} का भुगतान मिल गया है।",
            ],
            "kn": ["ನಿಮ್ಮ {amt} ಪಾವತಿ ಯಶಸ್ವಿಯಾಗಿದೆ."],
        },
    ),
    (
        "payment_failed",
        {
            "en": [
                "Your payment of {amt} has failed.",
                "The payment of {amt} did not go through.",
                "Unfortunately, your payment of {amt} was declined.",
            ],
            "hi": [
                "आपका {amt} का भुगतान विफल हो गया।",
                "{amt} का भुगतान नहीं हो पाया।",
            ],
            "kn": ["ನಿಮ್ಮ {amt} ಪಾವತಿ ವಿಫಲವಾಗಿದೆ."],
        },
    ),
    (
        "track_in_app",
        {
            "en": [
                "You can track your order in the app.",
                "You can follow your order's status in the app.",
                "Open the app to track your order.",
                "The app shows live tracking for your order.",
            ],
            "hi": [
                "आप ऐप में अपना ऑर्डर ट्रैक कर सकते हैं।",
                "Aap app mein apna order track kar sakte hain.",
            ],
            "kn": ["ನೀವು ಆ್ಯಪ್‌ನಲ್ಲಿ ನಿಮ್ಮ ಆರ್ಡರ್ ಟ್ರ್ಯಾಕ್ ಮಾಡಬಹುದು."],
        },
    ),
    (
        "order_out_for_delivery",
        {
            "en": [
                "Your order {order} is out for delivery.",
                "Your order {order} has gone out for delivery.",
                "Our delivery partner is on the way with order {order}.",
            ],
            "hi": [
                "आपका ऑर्डर {order} डिलीवरी के लिए निकल चुका है।",
                "Aapka order {order} out for delivery hai.",
            ],
            "kn": ["ನಿಮ್ಮ ಆರ್ಡರ್ {order} ಡೆಲಿವರಿಗೆ ಹೊರಟಿದೆ."],
        },
    ),
    (
        "ticket_created",
        {
            "en": [
                "Your ticket number is {ticket}.",
                "I have raised a ticket for you, and the number is {ticket}.",
                "Your complaint has been registered under ticket {ticket}.",
                "Please note your reference number {ticket}.",
            ],
            "hi": [
                "आपका टिकट नंबर {ticket} है।",
                "आपकी शिकायत टिकट {ticket} के तहत दर्ज कर ली गई है।",
            ],
            "kn": ["ನಿಮ್ಮ ಟಿಕೆಟ್ ಸಂಖ್ಯೆ {ticket}."],
        },
    ),
    (
        "order_arrive_on_date",
        {
            "en": [
                "Your order {order} will arrive on {date}.",
                "Order {order} is expected to reach you on {date}.",
                "The expected delivery date for order {order} is {date}.",
            ],
            "hi": [
                "आपका ऑर्डर {order} {date} को पहुँचेगा।",
                "ऑर्डर {order} की डिलीवरी {date} को होगी।",
            ],
            "kn": ["ನಿಮ್ಮ ಆರ್ಡರ್ {order} {date}ರಂದು ತಲುಪಲಿದೆ."],
        },
    ),
    (
        "order_arrive_by_date",
        {
            "en": [
                "You should receive order {order} by {date}.",
                "Order {order} will reach you by {date} at the latest.",
            ],
            "hi": ["ऑर्डर {order} आपको {date} तक मिल जाएगा।"],
        },
    ),
    (
        "order_delayed",
        {
            "en": [
                "Your order {order} has been delayed.",
                "I'm sorry, order {order} is running late.",
                "There is a delay with your order {order}.",
                "Unfortunately, order {order} will take longer than expected.",
            ],
            "hi": [
                "आपके ऑर्डर {order} में देरी हो गई है।",
                "माफ़ कीजिए, ऑर्डर {order} देर से पहुँचेगा।",
            ],
            "kn": ["ನಿಮ್ಮ ಆರ್ಡರ್ {order} ತಡವಾಗಿದೆ."],
        },
    ),
    (
        "apology",
        {
            "en": [
                "I'm sorry for the inconvenience.",
                "Apologies for the trouble.",
                "Sorry for the inconvenience caused.",
                "I understand, and I'm sorry about this.",
            ],
            "hi": ["असुविधा के लिए खेद है।", "परेशानी के लिए माफ़ी चाहते हैं।"],
            "kn": ["ಅನಾನುಕೂಲತೆಗಾಗಿ ಕ್ಷಮಿಸಿ."],
        },
    ),
    (
        "otp_sent",
        {
            "en": [
                "I have sent an OTP to your registered mobile number.",
                "An OTP has been sent to your registered number.",
                "We just sent a one-time password to your registered mobile number.",
            ],
            "hi": [
                "आपके रजिस्टर्ड मोबाइल नंबर पर OTP भेज दिया गया है।",
                "मैंने आपके नंबर पर OTP भेजा है।",
            ],
            "kn": ["ನಿಮ್ಮ ನೋಂದಾಯಿತ ಮೊಬೈಲ್ ಸಂಖ್ಯೆಗೆ OTP ಕಳುಹಿಸಲಾಗಿದೆ."],
        },
    ),
    (
        "order_cancelled",
        {
            "en": [
                "Your order {order} has been cancelled.",
                "Order {order} is now cancelled.",
                "We have cancelled your order {order}.",
                "I have cancelled order {order} for you.",
            ],
            "hi": [
                "आपका ऑर्डर {order} रद्द कर दिया गया है।",
                "ऑर्डर {order} कैंसल हो गया है।",
            ],
            "kn": ["ನಿಮ್ಮ ಆರ್ಡರ್ {order} ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ."],
        },
    ),
    (
        "refund_credit_days",
        {
            "en": [
                "The refund of {amt} will reach your account in {days} working days.",
                "Please allow {days} working days for the {amt} refund to reflect.",
            ],
            "hi": ["{amt} का रिफंड {days} कार्य दिवसों में आपके खाते में आ जाएगा।"],
        },
    ),
    (
        "refund_initiated",
        {
            "en": [
                "Your refund has been initiated.",
                "We have started processing your refund.",
                "The refund process has begun.",
            ],
            "hi": [
                "आपका रिफंड शुरू कर दिया गया है।",
                "Aapka refund initiate ho gaya hai.",
            ],
        },
    ),
    (
        "refund_rejected",
        {
            "en": [
                "Your refund request has been rejected.",
                "Unfortunately, we could not approve your refund.",
                "The refund request was declined.",
            ],
            "hi": [
                "आपका रिफंड अनुरोध अस्वीकार कर दिया गया है।",
                "माफ़ कीजिए, आपका रिफंड मंज़ूर नहीं हुआ।",
            ],
            "kn": ["ನಿಮ್ಮ ಮರುಪಾವತಿ ವಿನಂತಿ ತಿರಸ್ಕೃತವಾಗಿದೆ."],
        },
    ),
    (
        "callback_at_time",
        {
            "en": [
                "Our team will call you back at {time}.",
                "You will get a call from us at {time}.",
                "We have scheduled a callback for {time}.",
            ],
            "hi": [
                "हमारी टीम आपको {time} बजे कॉल करेगी।",
                "आपको {time} बजे हमारी तरफ से कॉल आएगा।",
            ],
            "kn": ["ನಮ್ಮ ತಂಡ ನಿಮಗೆ {time}ಕ್ಕೆ ಕರೆ ಮಾಡುತ್ತದೆ."],
        },
    ),
    (
        "specialist_call_at_time",
        {"en": ["A specialist will call you at {time}."]},
    ),
    (
        "callback_24h",
        {
            "en": [
                "Someone from our team will call you within 24 hours.",
                "You will hear back from us within 24 hours.",
            ],
            "hi": ["हमारी टीम 24 घंटे के अंदर आपको कॉल करेगी।"],
        },
    ),
    (
        "address_updated",
        {
            "en": [
                "Your delivery address has been updated.",
                "I have updated your address.",
                "The new address is now saved on your account.",
            ],
            "hi": ["आपका पता अपडेट कर दिया गया है।", "मैंने आपका पता बदल दिया है।"],
            "kn": ["ನಿಮ್ಮ ವಿಳಾಸವನ್ನು ನವೀಕರಿಸಲಾಗಿದೆ."],
        },
    ),
    (
        "order_arrive_today",
        {
            "en": [
                "Your order {order} will arrive today.",
                "Order {order} will be delivered today.",
            ],
            "hi": ["आपका ऑर्डर {order} आज पहुँचेगा।"],
            "kn": ["ನಿಮ್ಮ ಆರ್ಡರ್ {order} ಇಂದು ತಲುಪಲಿದೆ."],
        },
    ),
    (
        "order_arrive_tomorrow",
        {
            "en": [
                "Your order {order} will arrive tomorrow.",
                "Order {order} will be delivered tomorrow.",
            ],
            "hi": ["आपका ऑर्डर {order} कल पहुँचेगा।"],
            "kn": ["ನಿಮ್ಮ ಆರ್ಡರ್ {order} ನಾಳೆ ತಲುಪಲಿದೆ."],
        },
    ),
    (
        "order_shipped_today",
        {"en": ["Order {order} was shipped earlier today."]},
    ),
    (
        "bill_due_on_date",
        {
            "en": [
                "Your bill of {amt} is due on {date}.",
                "A payment of {amt} is due on {date}.",
                "Please pay your bill of {amt} by {date}.",
            ],
            "hi": [
                "आपका {amt} का बिल {date} को देय है।",
                "कृपया {date} तक {amt} का बिल भर दें।",
            ],
            "kn": ["ನಿಮ್ಮ {amt} ಬಿಲ್ {date}ರಂದು ಪಾವತಿಸಬೇಕು."],
        },
    ),
    (
        "order_arrive_in_days",
        {
            "en": [
                "Your order {order} will arrive in {days} days.",
                "Order {order} will be with you in {days} days.",
            ],
            "hi": ["आपका ऑर्डर {order} {days} दिनों में पहुँच जाएगा।"],
        },
    ),
    (
        "order_arrive_within_days",
        {
            "en": [
                "Order {order} should reach you within {days} days.",
                "You will get order {order} within {days} days.",
            ],
        },
    ),
    (
        "payment_pending",
        {
            "en": [
                "Your payment is still pending.",
                "The payment is being processed and has not been confirmed yet.",
                "We are yet to receive confirmation of your payment.",
            ],
            "hi": [
                "आपका भुगतान अभी प्रोसेस हो रहा है।",
                "Aapka payment abhi pending hai.",
            ],
        },
    ),
    (
        "otp_expired",
        {
            "en": ["The OTP has expired.", "That OTP is no longer valid."],
            "hi": [
                "OTP की समय सीमा समाप्त हो गई है।",
                "यह OTP अब मान्य नहीं है।",
            ],
        },
    ),
    (
        "otp_incorrect",
        {
            "en": ["The OTP you entered is incorrect.", "That OTP does not match."],
            "hi": ["आपने जो OTP डाला है वह गलत है।"],
        },
    ),
    (
        "otp_valid_minutes",
        {
            "en": [
                "The OTP is valid for {mins} minutes.",
                "This OTP will expire in {mins} minutes.",
            ],
            "hi": ["यह OTP {mins} मिनट के लिए मान्य है।"],
        },
    ),
    (
        "support_weekdays",
        {
            "en": [
                "Our support team is available on weekdays.",
                "You can reach our support team Monday to Friday.",
                "Our agents are available from Monday to Friday.",
            ],
            "hi": [
                "हमारी सपोर्ट टीम सोमवार से शुक्रवार तक उपलब्ध है।",
                "सपोर्ट टीम कार्यदिवसों में उपलब्ध रहती है।",
            ],
        },
    ),
    (
        "support_24x7",
        {
            "en": ["Our support team is available 24/7."],
            "hi": ["हमारी सपोर्ट टीम चौबीसों घंटे उपलब्ध है।"],
        },
    ),
    (
        "technician_visit",
        {
            "en": [
                "The technician will visit on {date} at {time}.",
                "A technician is scheduled to come on {date} at {time}.",
                "Your technician visit is booked for {date} at {time}.",
            ],
            "hi": ["तकनीशियन {date} को {time} बजे आएगा।"],
        },
    ),
    (
        "kyc_pending",
        {
            "en": [
                "Your KYC verification is still pending.",
                "We have not completed your KYC verification yet.",
                "Your KYC is under review.",
            ],
            "hi": ["आपका KYC सत्यापन अभी लंबित है।", "Aapka KYC abhi pending hai."],
            "kn": ["ನಿಮ್ಮ KYC ಪರಿಶೀಲನೆ ಇನ್ನೂ ಬಾಕಿ ಇದೆ."],
        },
    ),
    (
        "kyc_done",
        {
            "en": [
                "Your KYC verification is complete.",
                "Your KYC has been approved.",
            ],
            "hi": ["आपका KYC सत्यापन पूरा हो गया है।"],
        },
    ),
    (
        "password_reset_sent",
        {
            "en": [
                "I have sent a password reset link to your email.",
                "A link to reset your password has been emailed to you.",
                "Please check your email for the password reset link.",
            ],
            "hi": ["पासवर्ड रीसेट करने का लिंक आपके ईमेल पर भेज दिया गया है।"],
        },
    ),
    (
        "transfer_agent",
        {
            "en": [
                "Let me connect you to a human agent.",
                "I will transfer you to one of our agents.",
            ],
            "hi": ["मैं आपको हमारे एजेंट से जोड़ रहा हूँ।"],
            "kn": ["ನಾನು ನಿಮ್ಮನ್ನು ನಮ್ಮ ಏಜೆಂಟ್‌ಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ."],
        },
    ),
    (
        "cannot_help",
        {
            "en": [
                "I'm sorry, I can't help with that.",
                "Unfortunately, that is not something I can do.",
                "I am not able to help with that request.",
            ],
            "hi": ["माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता।"],
        },
    ),
]

# (meaning, weight, {language: [wordings]}). A {name} in the meaning makes every customer's sentence distinct.
OPENERS = [
    (
        "ack",
        40,
        {
            "en": ["Sure.", "Got it.", "Okay.", "Certainly.", "Alright."],
            "hi": ["ज़रूर।", "ठीक है।", "जी।"],
            "kn": ["ಸರಿ."],
        },
    ),
    (
        "greeting_thanks",
        25,
        {
            "en": [
                "Hello, thanks for calling.",
                "Hi, thanks for reaching out.",
                "Thank you for contacting us.",
            ],
            "hi": ["नमस्ते, कॉल करने के लिए धन्यवाद।", "हमसे संपर्क करने के लिए धन्यवाद।"],
            "kn": ["ನಮಸ್ಕಾರ, ಕರೆ ಮಾಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು."],
        },
    ),
    (
        "thanks_patience",
        15,
        {
            "en": [
                "Thank you for your patience.",
                "Thanks for waiting.",
                "I appreciate your patience.",
            ],
            "hi": ["धैर्य रखने के लिए धन्यवाद।", "इंतज़ार करने के लिए शुक्रिया।"],
            "kn": ["ನಿಮ್ಮ ತಾಳ್ಮೆಗೆ ಧನ್ಯವಾದಗಳು."],
        },
    ),
    (
        "thanks_patience:{name}",
        12,
        {
            "en": ["Thanks for your patience, {name}.", "Thank you for waiting, {name}."],
            "hi": ["धन्यवाद, {name} जी।"],
            "kn": ["ಧನ್ಯವಾದಗಳು, {name}."],
        },
    ),
    (
        "can_help:{name}",
        8,
        {
            "en": ["Hi {name}, I can help you with that."],
            "hi": ["{name} जी, मैं इसमें आपकी मदद कर सकता हूँ।"],
        },
    ),
]

CLOSERS = [
    (
        "anything_else",
        60,
        {
            "en": [
                "Is there anything else I can help you with?",
                "Can I help you with anything else?",
                "Is there something else you need?",
                "Anything else I can do for you?",
            ],
            "hi": [
                "क्या मैं आपकी और कोई मदद कर सकता हूँ?",
                "क्या कुछ और है जिसमें मैं मदद कर सकूँ?",
            ],
            "kn": ["ನಾನು ಇನ್ನೇನಾದರೂ ಸಹಾಯ ಮಾಡಬಹುದೇ?"],
        },
    ),
    (
        "goodbye",
        30,
        {
            "en": [
                "Thank you, have a great day!",
                "Thanks for calling, have a nice day!",
                "Take care and have a good day!",
            ],
            "hi": ["धन्यवाद, आपका दिन शुभ हो।", "आपका दिन मंगलमय हो।"],
            "kn": ["ಧನ್ಯವಾದಗಳು, ಶುಭ ದಿನ."],
        },
    ),
    (
        "goodbye:{name}",
        10,
        {
            "en": ["Have a great day, {name}!"],
            "hi": ["आपका दिन शुभ हो, {name} जी।"],
        },
    ),
]

# Free-form explanations: an LLM composes these per case, so exact repeats are rare.
DETAILS = {
    "en": (
        "I can see that your {product} was {status} on {date}, and {step}.",
        [
            "phone case",
            "wireless earbuds",
            "running shoes",
            "water bottle",
            "laptop bag",
            "smartwatch",
            "kurta",
            "mixer grinder",
            "bedsheet set",
            "backpack",
        ],
        [
            "delivered",
            "picked up for return",
            "handed to the courier",
            "returned to our warehouse",
            "marked as undeliverable",
        ],
        [
            "the refund will start once we inspect it",
            "you do not need to do anything else",
            "I have asked the courier to try again",
            "a replacement is being arranged",
            "I have flagged this to our logistics team",
            "you can raise a return from the app",
        ],
    ),
    "hi": (
        "आपका {product} {date} को {status}, और {step}।",
        ["फ़ोन कवर", "ईयरबड्स", "जूते", "बैग", "स्मार्टवॉच", "कुर्ता"],
        [
            "डिलीवर हो गया था",
            "वापस लौटा दिया गया था",
            "कूरियर को सौंप दिया गया था",
        ],
        [
            "जाँच के बाद रिफंड शुरू होगा",
            "आपको कुछ और करने की ज़रूरत नहीं है",
            "मैंने कूरियर से दोबारा प्रयास करने को कहा है",
            "रिप्लेसमेंट की व्यवस्था की जा रही है",
        ],
    ),
    "kn": (
        "ನಿಮ್ಮ {product} {status}, ಮತ್ತು {step}.",
        ["ಫೋನ್ ಕವರ್", "ಶೂಗಳು", "ಬ್ಯಾಗ್"],
        ["ತಲುಪಿಸಲಾಗಿದೆ", "ಹಿಂತಿರುಗಿಸಲಾಗಿದೆ"],
        [
            "ಮರುಪಾವತಿ ಶೀಘ್ರದಲ್ಲೇ ಪ್ರಾರಂಭವಾಗುತ್ತದೆ",
            "ಬದಲಿ ವ್ಯವಸ್ಥೆ ಮಾಡಲಾಗುತ್ತಿದೆ",
            "ನೀವು ಬೇರೇನೂ ಮಾಡಬೇಕಾಗಿಲ್ಲ",
        ],
    ),
}

NAMES = {
    "en": [
        "Priya", "Rahul", "Ananya", "Arjun", "Sneha", "Vikram", "Divya", "Karthik", "Meera", "Rohan",
        "Aisha", "Suresh", "Kavya", "Nikhil", "Pooja", "Amit", "Neha", "Sanjay", "Lakshmi", "Farhan",
    ],
    "hi": ["प्रिया", "राहुल", "अनन्या", "अर्जुन", "स्नेहा", "विक्रम", "दिव्या", "अमित", "नेहा", "संजय", "पूजा", "रोहन"],
    "kn": ["ಪ್ರಿಯಾ", "ರಾಹುಲ್", "ಅನನ್ಯಾ", "ಕಾರ್ತಿಕ್", "ಮೇಘನಾ", "ಸುರೇಶ್"],
}

MONTHS = {
    "en": ["January", "March", "April", "June", "August", "October", "December"],
    "hi": ["जनवरी", "मार्च", "अप्रैल", "जून", "अगस्त", "अक्टूबर", "दिसंबर"],
    "kn": ["ಜನವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಜೂನ್", "ಆಗಸ್ಟ್", "ಅಕ್ಟೋಬರ್", "ಡಿಸೆಂಬರ್"],
}

PRICES = ["199", "299", "349", "499", "599", "799", "999", "1,299", "1,499", "2,499", "3,999", "12,999"]


def fill_values(rng: random.Random, language: str) -> dict[str, str]:
    """Realistic values for every placeholder, written the way this language's traffic writes them."""
    order = (
        f"OD{rng.randint(10**9, 10**10 - 1)}"  # alphanumeric: no template can slot it
        if rng.random() < 0.25
        else str(rng.randint(100000, 999999))
    )

    price = rng.choice(PRICES)
    currency_forms = {
        "en": [f"₹{price}", f"Rs. {price}", f"Rs {price}", f"INR {price}", f"{price} rupees"],
        "hi": [f"₹{price}", f"{price} रुपये", f"Rs. {price}"],
        "kn": [f"₹{price}", f"{price} ರೂಪಾಯಿ"],
    }
    amt = rng.choice(currency_forms[language])

    day, month = rng.randint(1, 28), rng.randint(1, 12)
    if rng.random() < 0.7:
        date = f"{day:02d}/{month:02d}/2026"
    elif language == "kn":
        date = f"{rng.choice(MONTHS[language])} {day}"
    else:
        date = f"{day} {rng.choice(MONTHS[language])}"

    hour, minute = rng.randint(9, 19), rng.choice(["00", "15", "30", "45"])
    if language == "en":
        time = f"{(hour - 1) % 12 + 1}:{minute} {'AM' if hour < 12 else 'PM'}"
    else:
        time = f"{(hour - 1) % 12 + 1}:{minute}"

    return {
        "order": order,
        "amt": amt,
        "date": date,
        "time": time,
        "days": str(rng.randint(2, 7)),
        "mins": rng.choice(["5", "10"]),
        "ticket": str(rng.randint(10000, 99999)),
        "name": rng.choice(NAMES[language]),
    }


def say(rng, meaning: str, wordings: list[str], language: str) -> tuple[str, str]:
    """One sentence and its meaning label, with fresh values."""
    values = fill_values(rng, language)
    wording = zipf_choice(rng, wordings, s=1.0)
    return wording.format(**values), meaning.format(**values)


def pick_weighted(rng, groups, language):
    available = [g for g in groups if language in g[2]]
    meaning, _, wordings = rng.choices(available, weights=[g[1] for g in available])[0]
    return say(rng, meaning, wordings[language], language)


def pick_intent(rng, language, exclude: set[str]) -> tuple[str, str]:
    available = [(m, w) for m, w in INTENTS if language in w and m not in exclude]
    meaning, wordings = zipf_choice(rng, available, s=0.9)
    return say(rng, meaning, wordings[language], language)


def pick_detail(rng, language) -> tuple[str, str]:
    pattern, products, statuses, steps = DETAILS[language]
    product = zipf_choice(rng, products, s=0.7)
    status = zipf_choice(rng, statuses, s=0.7)
    step = zipf_choice(rng, steps, s=0.7)
    date = fill_values(rng, language)["date"]
    text = pattern.format(product=product, status=status, step=step, date=date)
    return text, f"detail:{product}|{status}|{step}"


def check_alignment(request: Request) -> None:
    """Every label must line up with exactly one sentence as the cache will split it."""
    sentences = split_sentences(normalize(request.text, request.language), request.language)
    if len(sentences) != len(request.meanings):
        raise ValueError(f"{len(sentences)} sentences, {len(request.meanings)} labels: {request.text}")


def generate_varied(n_requests: int = 8000, n_users: int = 2000, seed: int = 11) -> list[Request]:
    rng = random.Random(seed)
    languages, shares = zip(*LANGUAGE_SHARE.items())
    requests = []

    for _ in range(n_requests):
        language = rng.choices(languages, weights=shares)[0]
        parts = []

        if rng.random() < 0.5:
            parts.append(pick_weighted(rng, OPENERS, language))
        for _ in range(1 if rng.random() < 0.65 else 2):
            if rng.random() < 0.25:
                parts.append(pick_detail(rng, language))
            else:
                parts.append(pick_intent(rng, language, exclude={m for _, m in parts}))
        if rng.random() < 0.5:
            parts.append(pick_weighted(rng, CLOSERS, language))

        request = Request(
            user_id=f"user{rng.randrange(n_users)}",
            language=language,
            text=" ".join(t for t, _ in parts),
            meanings=[m for _, m in parts],
        )
        check_alignment(request)
        requests.append(request)

    return requests


# ---------------- Labels for the base dataset ----------------

# Meaning of every sentence the base generator can produce (its long-tail sentences are each unique).
BASE_MEANINGS = {
    "en": {
        "Hello, thanks for calling.": "greeting_thanks",
        "Hi!": "greeting_hi",
        "How can I help you today?": "greeting_help",
        "Welcome back.": "welcome_back",
        "Please hold while I check that for you.": "hold_checking",
        "Please wait while I look into that for you.": "hold_checking",
        "Your request has been received.": "request_received",
        "We have received your request.": "request_received",
        "You can track your order in the app.": "track_in_app",
        "The payment was successful.": "payment_success",
        "The payment was not successful.": "payment_failed",
        "Our support team is available on weekdays.": "support_weekdays",
        "Your request has been rejected.": "request_rejected",
        "Your order {n} has been shipped.": "order_shipped",
        "We have shipped your order {n}.": "order_shipped",
        "Your refund of Rs. {amt} has been processed.": "refund_processed",
        "Your ticket number is {n}.": "ticket_created",
        "Your bill of ₹{amt} is due on Friday.": "bill_due_friday",
        "Your order {n} will arrive today.": "order_arrive_today",
        "Your order {n} will arrive tomorrow.": "order_arrive_tomorrow",
        "Your order {n} will arrive in {d} days.": "order_arrive_in_days",
        "Your appointment is on {date}.": "appointment_on_date",
        "The technician will visit at {time}.": "technician_at_time",
        "Is there anything else I can help you with?": "anything_else",
        "Thank you, have a great day!": "goodbye",
    },
    "hi": {
        "नमस्ते, कॉल करने के लिए धन्यवाद।": "greeting_thanks",
        "मैं आपकी क्या मदद कर सकता हूँ?": "greeting_help",
        "आपका फिर से स्वागत है।": "welcome_back",
        "कृपया प्रतीक्षा करें, मैं जाँच कर रहा हूँ।": "hold_checking",
        "आपका अनुरोध प्राप्त हो गया है।": "request_received",
        "हमें आपका अनुरोध मिल गया है।": "request_received",
        "आप ऐप में अपना ऑर्डर ट्रैक कर सकते हैं।": "track_in_app",
        "भुगतान सफल रहा।": "payment_success",
        "भुगतान सफल नहीं रहा।": "payment_failed",
        "आपका ऑर्डर {n} भेज दिया गया है।": "order_shipped",
        "हमने आपका ऑर्डर {n} भेज दिया है।": "order_shipped",
        "आपका ₹{amt} का रिफंड प्रोसेस हो गया है।": "refund_processed",
        "आपका टिकट नंबर {n} है।": "ticket_created",
        "आपकी अपॉइंटमेंट {date} को है।": "appointment_on_date",
        "आपका ऑर्डर {n} आज पहुँचेगा।": "order_arrive_today",
        "आपका ऑर्डर {n} कल पहुँचेगा।": "order_arrive_tomorrow",
        "क्या मैं आपकी और कोई मदद कर सकता हूँ?": "anything_else",
        "धन्यवाद, आपका दिन शुभ हो।": "goodbye",
    },
}
BASE_SAMPLE_VALUES = {"n": "1234", "amt": "499", "d": "3", "date": "01/02/2026", "time": "9:30"}


def meaning_key(sentence: str) -> str:
    """What a sentence is cached and semantically matched as: its template if it has slots, else itself."""
    words = sentence.split()
    positions = find_variables(words)
    return make_template(words, positions)[0] if positions else sentence


def _base_lookup() -> dict[str, str]:
    lookup = {}
    for language, meanings in BASE_MEANINGS.items():
        for text, meaning in meanings.items():
            sentence = normalize(text.format(**BASE_SAMPLE_VALUES), language)
            lookup[meaning_key(sentence)] = meaning
    return lookup


def label_base(requests: list[Request]) -> list[Request]:
    """Attach meaning labels to base-generator requests."""
    lookup = _base_lookup()
    for r in requests:
        sentences = split_sentences(normalize(r.text, r.language), r.language)
        r.meanings = [lookup.get(meaning_key(s), f"tail:{s}") for s in sentences]
        for s, m in zip(sentences, r.meanings):
            if m.startswith("tail:") and not (s.startswith("Regarding") or "के बारे में" in s):
                raise ValueError(f"unlabeled base sentence: {s}")
    return requests


def interleave(a: list, b: list, rng: random.Random) -> list:
    """Merge two request streams at random, keeping each stream's own order."""
    merged, i, j = [], 0, 0
    while i < len(a) or j < len(b):
        if rng.random() < (len(a) - i) / (len(a) - i + len(b) - j):
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    return merged
