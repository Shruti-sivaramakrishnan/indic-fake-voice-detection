"""
Real interface translations for app.py -- English, Hindi, Tamil, Kannada.

Only interface chrome (labels, buttons, captions, headings) is translated.
`detected_language` in an AnalysisResult is a model output and is never
touched here -- it is displayed exactly as the pipeline returns it (currently
always "unknown", since there is no language classifier), regardless of which
interface language is selected. See the LANGUAGE_CAPTION string below, which
says so explicitly in the UI itself.

Translations are genuine, not placeholders, but were produced by an AI
assistant rather than a native-speaker reviewer -- worth a native-speaker
pass before this ships to real users, particularly for tone/register.
"""

from __future__ import annotations

LANGUAGES = ["English", "Hindi", "Tamil", "Kannada"]

TRANSLATIONS: dict[str, dict[str, str]] = {
    "app_title": {
        "English": "Voice Guard", "Hindi": "Voice Guard",
        "Tamil": "Voice Guard", "Kannada": "Voice Guard",
    },
    "app_tagline": {
        "English": "AI-generated voice detection for the calls that try to catch you off guard.",
        "Hindi": "उन कॉल्स के लिए एआई-जनित आवाज़ की पहचान, जो आपको चौंकाने की कोशिश करती हैं।",
        "Tamil": "உங்களை திடீரென ஏமாற்ற முயலும் அழைப்புகளுக்கான, AI-உருவாக்கிய குரல் கண்டறிதல்.",
        "Kannada": "ನಿಮ್ಮನ್ನು ದಿಢೀರನೆ ವಂಚಿಸಲು ಪ್ರಯತ್ನಿಸುವ ಕರೆಗಳಿಗಾಗಿ AI-ಉತ್ಪಾದಿತ ಧ್ವನಿ ಪತ್ತೆ.",
    },
    "landing_p1": {
        "English": "It starts with a call. A voice that sounds exactly like someone you trust, "
                   "a relative, a friend, asking for money, urgently, right now. It's a cloned "
                   "voice built from a few seconds of audio.",
        "Hindi": "यह एक कॉल से शुरू होता है। एक आवाज़ जो बिल्कुल किसी भरोसेमंद व्यक्ति जैसी लगती है, "
                 "कोई रिश्तेदार, कोई दोस्त, जो तुरंत, अभी पैसे मांग रहा है। यह कुछ सेकंड की ऑडियो से "
                 "बनाई गई क्लोन आवाज़ है।",
        "Tamil": "இது ஒரு அழைப்பில் இருந்து தொடங்குகிறது. நீங்கள் நம்பும் ஒருவரின் குரல் போலவே "
                 "ஒலிக்கும் ஒரு குரல், ஒரு உறவினர், ஒரு நண்பர், உடனடியாக, இப்போதே பணம் கேட்கிறார். "
                 "இது சில வினாடிகள் ஆடியோவிலிருந்து உருவாக்கப்பட்ட நகல் குரல்.",
        "Kannada": "ಇದು ಒಂದು ಕರೆಯಿಂದ ಪ್ರಾರಂಭವಾಗುತ್ತದೆ. ನೀವು ನಂಬುವ ಯಾರೋ ಒಬ್ಬರ ಧ್ವನಿಯಂತೆಯೇ ಕೇಳುವ "
                  "ಧ್ವನಿ, ಒಬ್ಬ ಸಂಬಂಧಿ, ಒಬ್ಬ ಸ್ನೇಹಿತ, ತಕ್ಷಣ, ಈಗಲೇ ಹಣ ಕೇಳುತ್ತಿದ್ದಾರೆ. ಇದು ಕೆಲವು "
                  "ಸೆಕೆಂಡುಗಳ ಆಡಿಯೊದಿಂದ ರಚಿಸಲಾದ ಕ್ಲೋನ್ ಮಾಡಿದ ಧ್ವನಿ.",
    },
    "landing_p2": {
        "English": "Voice Guard checks a recording in seconds and tells you whether it's real "
                   "or AI-generated, before you act on it.",
        "Hindi": "Voice Guard कुछ ही सेकंड में रिकॉर्डिंग जांचता है और बताता है कि यह असली है या "
                 "एआई-जनित, इससे पहले कि आप कोई कार्रवाई करें।",
        "Tamil": "Voice Guard சில வினாடிகளில் பதிவைச் சரிபார்த்து, நீங்கள் நடவடிக்கை எடுப்பதற்கு "
                 "முன், அது உண்மையானதா அல்லது AI-உருவாக்கியதா என்று சொல்கிறது.",
        "Kannada": "ನೀವು ಕ್ರಮ ಕೈಗೊಳ್ಳುವ ಮೊದಲು, Voice Guard ಕೆಲವೇ ಸೆಕೆಂಡುಗಳಲ್ಲಿ ರೆಕಾರ್ಡಿಂಗ್ ಅನ್ನು "
                  "ಪರಿಶೀಲಿಸಿ ಅದು ನಿಜವೋ ಅಥವಾ AI-ಉತ್ಪಾದಿತವೋ ಎಂದು ಹೇಳುತ್ತದೆ.",
    },
    "cta_button": {
        "English": "Check a voice message", "Hindi": "आवाज़ का संदेश जांचें",
        "Tamil": "குரல் செய்தியை சரிபார்க்கவும்", "Kannada": "ಧ್ವನಿ ಸಂದೇಶವನ್ನು ಪರಿಶೀಲಿಸಿ",
    },
    "screen2_heading": {
        "English": "Upload a voice clip", "Hindi": "आवाज़ की क्लिप अपलोड करें",
        "Tamil": "குரல் கிளிப்பைப் பதிவேற்றவும்", "Kannada": "ಧ್ವನಿ ಕ್ಲಿಪ್ ಅನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    },
    "interface_language_label": {
        "English": "Interface language", "Hindi": "इंटरफ़ेस भाषा",
        "Tamil": "இடைமுக மொழி", "Kannada": "ಇಂಟರ್ಫೇಸ್ ಭಾಷೆ",
    },
    "language_caption": {
        "English": "This only changes the interface text. It is not sent to the model, so "
                   "detected language below will still read \"unknown\".",
        "Hindi": "यह केवल इंटरफ़ेस के टेक्स्ट को बदलता है। यह मॉडल को नहीं भेजा जाता, इसलिए नीचे "
                 "पहचानी गई भाषा \"unknown\" ही दिखेगी।",
        "Tamil": "இது இடைமுக உரையை மட்டுமே மாற்றுகிறது. இது மாதிரிக்கு அனுப்பப்படுவதில்லை, எனவே "
                 "கீழே கண்டறியப்பட்ட மொழி \"unknown\" எனவே காட்டப்படும்.",
        "Kannada": "ಇದು ಇಂಟರ್ಫೇಸ್ ಪಠ್ಯವನ್ನು ಮಾತ್ರ ಬದಲಾಯಿಸುತ್ತದೆ. ಇದನ್ನು ಮಾದರಿಗೆ ಕಳುಹಿಸಲಾಗುವುದಿಲ್ಲ, "
                  "ಆದ್ದರಿಂದ ಕೆಳಗಿನ ಪತ್ತೆಯಾದ ಭಾಷೆ ಇನ್ನೂ \"unknown\" ಎಂದೇ ತೋರಿಸುತ್ತದೆ.",
    },
    "upload_label": {
        "English": "Upload an audio clip", "Hindi": "एक ऑडियो क्लिप अपलोड करें",
        "Tamil": "ஒரு ஆடியோ கிளிப்பைப் பதிவேற்றவும்", "Kannada": "ಒಂದು ಆಡಿಯೊ ಕ್ಲಿಪ್ ಅನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    },
    "analyze_button": {
        "English": "Analyze this clip", "Hindi": "इस क्लिप का विश्लेषण करें",
        "Tamil": "இந்த கிளிப்பை பகுப்பாய்வு செய்யவும்", "Kannada": "ಈ ಕ್ಲಿಪ್ ಅನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
    },
    "no_file_title": {
        "English": "No file selected", "Hindi": "कोई फ़ाइल चयनित नहीं है",
        "Tamil": "கோப்பு எதுவும் தேர்ந்தெடுக்கப்படவில்லை", "Kannada": "ಯಾವುದೇ ಫೈಲ್ ಆಯ್ಕೆಯಾಗಿಲ್ಲ",
    },
    "no_file_body": {
        "English": "Upload an audio clip above before analyzing.",
        "Hindi": "विश्लेषण करने से पहले ऊपर एक ऑडियो क्लिप अपलोड करें।",
        "Tamil": "பகுப்பாய்வு செய்வதற்கு முன் மேலே ஒரு ஆடியோ கிளிப்பைப் பதிவேற்றவும்.",
        "Kannada": "ವಿಶ್ಲೇಷಿಸುವ ಮೊದಲು ಮೇಲೆ ಒಂದು ಆಡಿಯೊ ಕ್ಲಿಪ್ ಅನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
    },
    "verdict_real_title": {
        "English": "Likely Real", "Hindi": "संभवतः असली",
        "Tamil": "உண்மையாக இருக்கலாம்", "Kannada": "ಬಹುಶಃ ನೈಜ",
    },
    "verdict_real_body": {
        "English": "This clip matches the acoustic pattern of a genuine human voice.",
        "Hindi": "यह क्लिप एक वास्तविक मानव आवाज़ के ध्वनि पैटर्न से मेल खाती है।",
        "Tamil": "இந்த கிளிப் ஒரு உண்மையான மனித குரலின் ஒலி வடிவத்துடன் பொருந்துகிறது.",
        "Kannada": "ಈ ಕ್ಲಿಪ್ ನಿಜವಾದ ಮಾನವ ಧ್ವನಿಯ ಧ್ವನಿ ಮಾದರಿಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತದೆ.",
    },
    "verdict_fake_title": {
        "English": "Likely Fake, AI-generated", "Hindi": "संभवतः फ़र्ज़ी, एआई-जनित",
        "Tamil": "போலியாக இருக்கலாம், AI-உருவாக்கியது", "Kannada": "ಬಹುಶಃ ನಕಲಿ, AI-ಉತ್ಪಾದಿತ",
    },
    "verdict_fake_body": {
        "English": "This clip shows patterns consistent with synthetic speech. Treat any "
                   "request made in it with caution.",
        "Hindi": "यह क्लिप सिंथेटिक स्पीच से मेल खाने वाले पैटर्न दिखाती है। इसमें की गई किसी भी "
                 "मांग को सावधानी से लें।",
        "Tamil": "இந்த கிளிப் செயற்கை பேச்சுடன் பொருந்தும் வடிவங்களைக் காட்டுகிறது. இதில் "
                 "செய்யப்படும் எந்தக் கோரிக்கையையும் எச்சரிக்கையுடன் கையாளவும்.",
        "Kannada": "ಈ ಕ್ಲಿಪ್ ಕೃತಕ ಭಾಷಣಕ್ಕೆ ಹೊಂದಿಕೆಯಾಗುವ ಮಾದರಿಗಳನ್ನು ತೋರಿಸುತ್ತದೆ. ಇದರಲ್ಲಿ ಮಾಡಿದ "
                  "ಯಾವುದೇ ವಿನಂತಿಯನ್ನು ಎಚ್ಚರಿಕೆಯಿಂದ ಪರಿಗಣಿಸಿ.",
    },
    "verdict_uncertain_title": {
        "English": "Uncertain, inconclusive", "Hindi": "अनिश्चित, अनिर्णायक",
        "Tamil": "நிச்சயமற்றது, முடிவெடுக்க முடியாதது", "Kannada": "ಅನಿಶ್ಚಿತ, ನಿರ್ಣಾಯಕವಲ್ಲ",
    },
    "verdict_uncertain_body": {
        "English": "The model could not confidently call this real or fake. Do not rely on "
                   "this result alone, verify the caller another way.",
        "Hindi": "मॉडल विश्वास के साथ यह तय नहीं कर सका कि यह असली है या फ़र्ज़ी। केवल इस नतीजे "
                 "पर भरोसा न करें, कॉल करने वाले की पहचान किसी और तरीके से भी सुनिश्चित करें।",
        "Tamil": "இது உண்மையானதா அல்லது போலியானதா என்று மாதிரியால் நம்பிக்கையுடன் சொல்ல "
                 "முடியவில்லை. இந்த முடிவை மட்டும் நம்பாதீர்கள், அழைப்பவரை வேறு வழியில் "
                 "உறுதிப்படுத்தவும்.",
        "Kannada": "ಇದು ನೈಜವೋ ಅಥವಾ ನಕಲಿಯೋ ಎಂದು ಮಾದರಿಗೆ ವಿಶ್ವಾಸದಿಂದ ಹೇಳಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಈ "
                  "ಫಲಿತಾಂಶವನ್ನು ಮಾತ್ರ ನಂಬಬೇಡಿ, ಕರೆ ಮಾಡಿದವರನ್ನು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.",
    },
    "verdict_unknown_title": {
        "English": "Unrecognized result", "Hindi": "अपरिचित परिणाम",
        "Tamil": "அடையாளம் தெரியாத முடிவு", "Kannada": "ಗುರುತಿಸಲಾಗದ ಫಲಿತಾಂಶ",
    },
    "error_title": {
        "English": "Something went wrong", "Hindi": "कुछ गड़बड़ हो गई",
        "Tamil": "ஏதோ தவறு நடந்தது", "Kannada": "ಏನೋ ತಪ್ಪಾಗಿದೆ",
    },
    "meta_confidence": {
        "English": "Confidence", "Hindi": "विश्वास स्तर",
        "Tamil": "நம்பகத்தன்மை", "Kannada": "ವಿಶ್ವಾಸಾರ್ಹತೆ",
    },
    "meta_detected_language": {
        "English": "Detected language", "Hindi": "पहचानी गई भाषा",
        "Tamil": "கண்டறியப்பட்ட மொழி", "Kannada": "ಪತ್ತೆಯಾದ ಭಾಷೆ",
    },
    "meta_language_eer": {
        "English": "Language EER", "Hindi": "भाषा EER",
        "Tamil": "மொழி EER", "Kannada": "ಭಾಷೆ EER",
    },
    "lang_eer_unavailable": {
        "English": "not available yet", "Hindi": "अभी उपलब्ध नहीं",
        "Tamil": "இன்னும் கிடைக்கவில்லை", "Kannada": "ಇನ್ನೂ ಲಭ್ಯವಿಲ್ಲ",
    },
    "technical_details": {
        "English": "Technical details", "Hindi": "तकनीकी विवरण",
        "Tamil": "தொழில்நுட்ப விவரங்கள்", "Kannada": "ತಾಂತ್ರಿಕ ವಿವರಗಳು",
    },
    "pdf_heading": {
        "English": "PDF Evidence Report", "Hindi": "PDF साक्ष्य रिपोर्ट",
        "Tamil": "PDF சான்று அறிக்கை", "Kannada": "PDF ಪುರಾವೆ ವರದಿ",
    },
    "pdf_body": {
        "English": "Generate a shareable PDF with the verdict, confidence, and technical "
                   "details for this clip.",
        "Hindi": "इस क्लिप के लिए नतीजे, विश्वास स्तर और तकनीकी विवरण के साथ एक साझा करने योग्य "
                 "PDF बनाएं।",
        "Tamil": "இந்த கிளிப்பிற்கான முடிவு, நம்பகத்தன்மை மற்றும் தொழில்நுட்ப விவரங்களுடன் "
                 "பகிரக்கூடிய PDF ஒன்றை உருவாக்கவும்.",
        "Kannada": "ಈ ಕ್ಲಿಪ್‌ಗಾಗಿ ತೀರ್ಪು, ವಿಶ್ವಾಸಾರ್ಹತೆ ಮತ್ತು ತಾಂತ್ರಿಕ ವಿವರಗಳೊಂದಿಗೆ "
                  "ಹಂಚಿಕೊಳ್ಳಬಹುದಾದ PDF ಅನ್ನು ರಚಿಸಿ.",
    },
    "pdf_button": {
        "English": "Generate PDF Evidence Report", "Hindi": "PDF साक्ष्य रिपोर्ट बनाएं",
        "Tamil": "PDF சான்று அறிக்கையை உருவாக்கவும்", "Kannada": "PDF ಪುರಾವೆ ವರದಿಯನ್ನು ರಚಿಸಿ",
    },
    "coming_soon_badge": {
        "English": "Coming soon", "Hindi": "जल्द आ रहा है",
        "Tamil": "விரைவில் வருகிறது", "Kannada": "ಶೀಘ್ರದಲ್ಲೇ ಬರಲಿದೆ",
    },
    "whatsapp_section_heading": {
        "English": "Report scam calls directly", "Hindi": "फ़र्ज़ी कॉल की सीधे शिकायत करें",
        "Tamil": "மோசடி அழைப்புகளை நேரடியாக புகாரளிக்கவும்",
        "Kannada": "ಮೋಸದ ಕರೆಗಳನ್ನು ನೇರವಾಗಿ ವರದಿ ಮಾಡಿ",
    },
    "whatsapp_title": {
        "English": "WhatsApp Bot", "Hindi": "WhatsApp बॉट",
        "Tamil": "WhatsApp போட்", "Kannada": "WhatsApp ಬಾಟ್",
    },
    "whatsapp_body": {
        "English": "Forward a suspicious voice note straight to our WhatsApp number and get "
                   "a verdict back in seconds, no need to open this page.",
        "Hindi": "किसी संदिग्ध वॉइस नोट को सीधे हमारे WhatsApp नंबर पर फ़ॉरवर्ड करें और सेकंडों "
                 "में नतीजा पाएं, इस पेज को खोलने की ज़रूरत नहीं।",
        "Tamil": "சந்தேகத்திற்குரிய குரல் குறிப்பை நேரடியாக எங்கள் WhatsApp எண்ணுக்கு அனுப்பி, "
                 "சில வினாடிகளில் முடிவைப் பெறுங்கள், இந்தப் பக்கத்தைத் திறக்க வேண்டிய "
                 "அவசியமில்லை.",
        "Kannada": "ಅನುಮಾನಾಸ್ಪದ ಧ್ವನಿ ಸಂದೇಶವನ್ನು ನೇರವಾಗಿ ನಮ್ಮ WhatsApp ಸಂಖ್ಯೆಗೆ ಫಾರ್ವರ್ಡ್ ಮಾಡಿ "
                  "ಮತ್ತು ಕೆಲವೇ ಸೆಕೆಂಡುಗಳಲ್ಲಿ ಫಲಿತಾಂಶ ಪಡೆಯಿರಿ, ಈ ಪುಟವನ್ನು ತೆರೆಯುವ ಅಗತ್ಯವಿಲ್ಲ.",
    },
    "live_call_section_heading": {
        "English": "Get help during a call", "Hindi": "कॉल के दौरान मदद पाएं",
        "Tamil": "அழைப்பின் போது உதவி பெறுங்கள்", "Kannada": "ಕರೆಯ ಸಮಯದಲ್ಲಿ ಸಹಾಯ ಪಡೆಯಿರಿ",
    },
    "live_call_title": {
        "English": "Live Call Companion", "Hindi": "लाइव कॉल साथी",
        "Tamil": "நேரடி அழைப்பு துணை", "Kannada": "ಲೈವ್ ಕಾಲ್ ಕಂಪ್ಯಾನಿಯನ್",
    },
    "live_call_body": {
        "English": "A companion that listens alongside you during a live call and warns you "
                   "in real time if the voice looks synthetic.",
        "Hindi": "एक साथी जो लाइव कॉल के दौरान आपके साथ सुनता है और अगर आवाज़ सिंथेटिक लगे तो "
                 "आपको तुरंत आगाह करता है।",
        "Tamil": "நேரடி அழைப்பின் போது உங்களுடன் சேர்ந்து கேட்டு, குரல் செயற்கையானதாகத் "
                 "தோன்றினால் உடனடியாக உங்களை எச்சரிக்கும் ஒரு துணை.",
        "Kannada": "ಲೈವ್ ಕರೆಯ ಸಮಯದಲ್ಲಿ ನಿಮ್ಮೊಂದಿಗೆ ಆಲಿಸುವ ಮತ್ತು ಧ್ವನಿ ಕೃತಕವಾಗಿ ಕಂಡುಬಂದರೆ ನಿಜ "
                  "ಸಮಯದಲ್ಲಿ ನಿಮ್ಮನ್ನು ಎಚ್ಚರಿಸುವ ಸಂಗಾತಿ.",
    },
    "back_arrow_help": {
        "English": "Back", "Hindi": "वापस", "Tamil": "பின்செல்", "Kannada": "ಹಿಂದೆ",
    },
    "next_arrow_help": {
        "English": "Next", "Hindi": "आगे", "Tamil": "அடுத்து", "Kannada": "ಮುಂದೆ",
    },
}


def t(key: str, lang: str) -> str:
    """Look up a translated UI string. Falls back to English, then the raw key,
    so a missing translation degrades to readable text instead of a crash."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("English") or key
