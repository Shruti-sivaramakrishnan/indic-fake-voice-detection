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

# Focused analysis-screen copy. These strings deliberately call the output a
# model score/automated analysis rather than proof or a calibrated probability.
TRANSLATIONS.update({
    "how_it_works_title": {
        "English": "Analyze a voice recording",
        "Hindi": "वॉइस रिकॉर्डिंग का विश्लेषण करें",
        "Tamil": "குரல் பதிவைப் பகுப்பாய்வு செய்யுங்கள்",
        "Kannada": "ಧ್ವನಿ ರೆಕಾರ್ಡಿಂಗ್ ಅನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
    },
    "how_it_works_body": {
        "English": "Upload 1-30 seconds of clear speech. Voice Guard returns a cautious three-way result: likely genuine, inconclusive, or likely synthetic.",
        "Hindi": "1-30 सेकंड की साफ़ आवाज़ अपलोड करें। Voice Guard सावधानी से तीन में से एक नतीजा देता है: संभवतः असली, अनिर्णायक या संभवतः सिंथेटिक।",
        "Tamil": "1-30 வினாடிகள் தெளிவான பேச்சைப் பதிவேற்றவும். Voice Guard உண்மையாக இருக்கலாம், முடிவற்றது அல்லது செயற்கையாக இருக்கலாம் என்ற மூன்று முடிவுகளில் ஒன்றைக் காட்டும்.",
        "Kannada": "1-30 ಸೆಕೆಂಡುಗಳ ಸ್ಪಷ್ಟ ಮಾತನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ. Voice Guard ಬಹುಶಃ ನೈಜ, ಅನಿಶ್ಚಿತ ಅಥವಾ ಬಹುಶಃ ಕೃತಕ ಎಂಬ ಮೂರು ಫಲಿತಾಂಶಗಳಲ್ಲಿ ಒಂದನ್ನು ನೀಡುತ್ತದೆ.",
    },
    "upload_help": {
        "English": "WAV, MP3, FLAC, OGG, or M4A; maximum 25 MB and 30 seconds.",
        "Hindi": "WAV, MP3, FLAC, OGG या M4A; अधिकतम 25 MB और 30 सेकंड।",
        "Tamil": "WAV, MP3, FLAC, OGG அல்லது M4A; அதிகபட்சம் 25 MB மற்றும் 30 வினாடிகள்.",
        "Kannada": "WAV, MP3, FLAC, OGG ಅಥವಾ M4A; ಗರಿಷ್ಠ 25 MB ಮತ್ತು 30 ಸೆಕೆಂಡುಗಳು.",
    },
    "analyzing_status": {
        "English": "Loading the model and analyzing the selected clip...",
        "Hindi": "मॉडल लोड करके चुनी गई क्लिप का विश्लेषण हो रहा है...",
        "Tamil": "மாதிரி ஏற்றப்பட்டு தேர்ந்தெடுத்த கிளிப் பகுப்பாய்வு செய்யப்படுகிறது...",
        "Kannada": "ಮಾದರಿಯನ್ನು ಲೋಡ್ ಮಾಡಿ ಆಯ್ಕೆ ಮಾಡಿದ ಕ್ಲಿಪ್ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
    },
    "analysis_failed_body": {
        "English": "The clip could not be analyzed. Check the audio format and try again; technical details were logged privately.",
        "Hindi": "क्लिप का विश्लेषण नहीं हो सका। ऑडियो फ़ॉर्मैट जांचकर फिर कोशिश करें; तकनीकी विवरण निजी लॉग में दर्ज हैं।",
        "Tamil": "கிளிப்பை பகுப்பாய்வு செய்ய முடியவில்லை. ஆடியோ வடிவத்தைச் சரிபார்த்து மீண்டும் முயலவும்; தொழில்நுட்ப விவரங்கள் தனிப்பட்ட பதிவில் உள்ளன.",
        "Kannada": "ಕ್ಲಿಪ್ ಅನ್ನು ವಿಶ್ಲೇಷಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಆಡಿಯೊ ಸ್ವರೂಪ ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ; ತಾಂತ್ರಿಕ ವಿವರಗಳು ಖಾಸಗಿ ಲಾಗ್‌ನಲ್ಲಿ ಇವೆ.",
    },
    "ai_likelihood_score": {
        "English": "AI-likelihood model score",
        "Hindi": "एआई-संभावना मॉडल स्कोर",
        "Tamil": "AI சாத்திய மாதிரி மதிப்பெண்",
        "Kannada": "AI-ಸಂಭವನೀಯತೆ ಮಾದರಿ ಅಂಕ",
    },
    "score_explanation": {
        "English": "This score is the model's temperature-scaled output, not a guaranteed real-world probability.",
        "Hindi": "यह स्कोर मॉडल का तापमान-स्केल किया गया आउटपुट है, वास्तविक दुनिया की गारंटीकृत संभावना नहीं।",
        "Tamil": "இந்த மதிப்பெண் மாதிரியின் temperature-scaled வெளியீடு; இது உறுதியான நிஜ உலக நிகழ்தகவு அல்ல.",
        "Kannada": "ಈ ಅಂಕವು ಮಾದರಿಯ temperature-scaled ಫಲಿತಾಂಶ; ಇದು ಖಚಿತ ನೈಜ-ಜಗತ್ತಿನ ಸಂಭವನೀಯತೆ ಅಲ್ಲ.",
    },
    "likely_genuine_end": {
        "English": "0% - more genuine-like",
        "Hindi": "0% - असली आवाज़ जैसा",
        "Tamil": "0% - உண்மைக் குரல் போன்றது",
        "Kannada": "0% - ನೈಜ ಧ್ವನಿಯಂತೆ",
    },
    "likely_synthetic_end": {
        "English": "100% - more synthetic-like",
        "Hindi": "100% - सिंथेटिक आवाज़ जैसा",
        "Tamil": "100% - செயற்கைக் குரல் போன்றது",
        "Kannada": "100% - ಕೃತಕ ಧ್ವನಿಯಂತೆ",
    },
    "duration_label": {
        "English": "Duration", "Hindi": "अवधि", "Tamil": "கால அளவு", "Kannada": "ಅವಧಿ",
    },
    "file_label": {
        "English": "Selected file", "Hindi": "चुनी गई फ़ाइल", "Tamil": "தேர்ந்தெடுத்த கோப்பு", "Kannada": "ಆಯ್ಕೆ ಮಾಡಿದ ಫೈಲ್",
    },
    "size_label": {
        "English": "File size", "Hindi": "फ़ाइल का आकार", "Tamil": "கோப்பு அளவு", "Kannada": "ಫೈಲ್ ಗಾತ್ರ",
    },
    "safety_action_title": {
        "English": "Use the result as a signal, not a final decision",
        "Hindi": "नतीजे को संकेत मानें, अंतिम निर्णय नहीं",
        "Tamil": "முடிவை ஒரு சுட்டியாக மட்டும் பயன்படுத்துங்கள்; இறுதி தீர்ப்பாக அல்ல",
        "Kannada": "ಫಲಿತಾಂಶವನ್ನು ಸೂಚನೆಯಾಗಿ ಬಳಸಿ, ಅಂತಿಮ ನಿರ್ಧಾರವಾಗಿ ಅಲ್ಲ",
    },
    "safety_action_body": {
        "English": "For any important or urgent request, confirm the speaker through a separate contact method you already trust.",
        "Hindi": "किसी भी महत्वपूर्ण या तत्काल अनुरोध के लिए, किसी दूसरे भरोसेमंद संपर्क माध्यम से वक्ता की पुष्टि करें।",
        "Tamil": "முக்கியமான அல்லது அவசரமான கோரிக்கைக்கு, நீங்கள் ஏற்கனவே நம்பும் வேறு தொடர்பு வழியில் பேசுபவரை உறுதிப்படுத்துங்கள்.",
        "Kannada": "ಯಾವುದೇ ಪ್ರಮುಖ ಅಥವಾ ತುರ್ತು ವಿನಂತಿಗೆ, ನೀವು ಈಗಾಗಲೇ ನಂಬುವ ಪ್ರತ್ಯೇಕ ಸಂಪರ್ಕ ವಿಧಾನದಿಂದ ಮಾತನಾಡುವವರನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.",
    },
    "permanent_disclaimer": {
        "English": "Automated analysis can be wrong, especially with noise, compression, unseen generators, very short speech, or playback recordings. This is not proof of authenticity or deception.",
        "Hindi": "स्वचालित विश्लेषण गलत हो सकता है, खासकर शोर, कंप्रेशन, नए जनरेटर, बहुत छोटी आवाज़ या प्लेबैक रिकॉर्डिंग में। यह असलियत या धोखे का प्रमाण नहीं है।",
        "Tamil": "சத்தம், சுருக்கம், காணாத உருவாக்கிகள், மிகக் குறுகிய பேச்சு அல்லது மீண்டும் பதிவு செய்த ஆடியோவில் தானியங்கி பகுப்பாய்வு தவறாக இருக்கலாம். இது உண்மை அல்லது மோசடிக்கான சான்று அல்ல.",
        "Kannada": "ಶಬ್ದ, ಕಂಪ್ರೆಷನ್, ಕಾಣದ ಜನರೇಟರ್‌ಗಳು, ಅತಿ ಚಿಕ್ಕ ಮಾತು ಅಥವಾ ಪ್ಲೇಬ್ಯಾಕ್ ರೆಕಾರ್ಡಿಂಗ್‌ನಲ್ಲಿ ಸ್ವಯಂಚಾಲಿತ ವಿಶ್ಲೇಷಣೆ ತಪ್ಪಾಗಬಹುದು. ಇದು ನೈಜತೆ ಅಥವಾ ಮೋಸದ ಪುರಾವೆಯಲ್ಲ.",
    },
    "raw_score_label": {
        "English": "Raw model logit", "Hindi": "कच्चा मॉडल लॉजिट", "Tamil": "மூல மாதிரி logit", "Kannada": "ಮೂಲ ಮಾದರಿ logit",
    },
    "sample_rate_label": {
        "English": "Input sample rate", "Hindi": "इनपुट सैंपल रेट", "Tamil": "உள்ளீட்டு sample rate", "Kannada": "ಇನ್‌ಪುಟ್ sample rate",
    },
    "policy_label": {
        "English": "Decision policy", "Hindi": "निर्णय नीति", "Tamil": "முடிவு கொள்கை", "Kannada": "ನಿರ್ಧಾರ ನೀತಿ",
    },
    "quality_warning_clipping": {
        "English": "The recording may be clipped. A cleaner recording can produce a more reliable score.",
        "Hindi": "रिकॉर्डिंग क्लिप हो सकती है। साफ़ रिकॉर्डिंग अधिक भरोसेमंद स्कोर दे सकती है।",
        "Tamil": "பதிவு clipping ஆகியிருக்கலாம். தெளிவான பதிவு அதிக நம்பகமான மதிப்பெண் தரலாம்.",
        "Kannada": "ರೆಕಾರ್ಡಿಂಗ್ ಕ್ಲಿಪ್ ಆಗಿರಬಹುದು. ಸ್ವಚ್ಛ ರೆಕಾರ್ಡಿಂಗ್ ಹೆಚ್ಚು ವಿಶ್ವಾಸಾರ್ಹ ಅಂಕ ನೀಡಬಹುದು.",
    },
    "pdf_error": {
        "English": "The automated analysis report could not be generated.",
        "Hindi": "स्वचालित विश्लेषण रिपोर्ट नहीं बन सकी।",
        "Tamil": "தானியங்கி பகுப்பாய்வு அறிக்கையை உருவாக்க முடியவில்லை.",
        "Kannada": "ಸ್ವಯಂಚಾಲಿತ ವಿಶ್ಲೇಷಣಾ ವರದಿಯನ್ನು ರಚಿಸಲಾಗಲಿಲ್ಲ.",
    },
    "privacy_note": {
        "English": "Privacy: this demo processes the uploaded clip in the running Kaggle/Streamlit session. Do not upload confidential audio to a public deployment.",
        "Hindi": "गोपनीयता: यह डेमो अपलोड की गई क्लिप को चल रहे Kaggle/Streamlit सत्र में प्रोसेस करता है। सार्वजनिक डिप्लॉयमेंट पर गोपनीय ऑडियो अपलोड न करें।",
        "Tamil": "தனியுரிமை: பதிவேற்றிய கிளிப் இயங்கும் Kaggle/Streamlit அமர்வில் செயலாக்கப்படுகிறது. பொது deployment-இல் ரகசிய ஆடியோவைப் பதிவேற்ற வேண்டாம்.",
        "Kannada": "ಗೌಪ್ಯತೆ: ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಕ್ಲಿಪ್ ನಡೆಯುತ್ತಿರುವ Kaggle/Streamlit ಸೆಷನ್‌ನಲ್ಲಿ ಪ್ರಕ್ರಿಯೆಯಾಗುತ್ತದೆ. ಸಾರ್ವಜನಿಕ deployment‌ಗೆ ಗೌಪ್ಯ ಆಡಿಯೊ ಅಪ್‌ಲೋಡ್ ಮಾಡಬೇಡಿ.",
    },
    "nav_status": {
        "English": "Research preview",
        "Hindi": "रिसर्च प्रीव्यू",
        "Tamil": "ஆராய்ச்சி முன்னோட்டம்",
        "Kannada": "ಸಂಶೋಧನಾ ಪೂರ್ವವೀಕ್ಷಣೆ",
    },
    "hero_eyebrow": {
        "English": "Multilingual voice analysis",
        "Hindi": "बहुभाषी वॉइस विश्लेषण",
        "Tamil": "பலமொழி குரல் பகுப்பாய்வு",
        "Kannada": "ಬಹುಭಾಷಾ ಧ್ವನಿ ವಿಶ್ಲೇಷಣೆ",
    },
    "hero_title_main": {
        "English": "Is this voice human or",
        "Hindi": "यह आवाज़ इंसानी है या",
        "Tamil": "இந்தக் குரல் மனிதருடையதா அல்லது",
        "Kannada": "ಈ ಧ್ವನಿ ಮಾನವನದೋ ಅಥವಾ",
    },
    "hero_title_accent": {
        "English": "AI-generated?",
        "Hindi": "एआई से बनाई गई?",
        "Tamil": "AI உருவாக்கியதா?",
        "Kannada": "AI ರಚಿಸಿದ್ದೇ?",
    },
    "hero_body": {
        "English": "Upload a short recording and review the acoustic signals detected by Voice Guard. You receive a cautious result with the score and limitations clearly explained.",
        "Hindi": "एक छोटी रिकॉर्डिंग अपलोड करें और Voice Guard द्वारा पहचाने गए ध्वनिक संकेत देखें। आपको स्कोर और सीमाओं की साफ़ व्याख्या के साथ सावधानीपूर्ण नतीजा मिलेगा।",
        "Tamil": "ஒரு குறுகிய பதிவைப் பதிவேற்றி, Voice Guard கண்டறியும் ஒலியியல் சுட்டிகளைப் பாருங்கள். மதிப்பெண்ணும் வரம்புகளும் தெளிவாக விளக்கப்பட்ட எச்சரிக்கையான முடிவைப் பெறுவீர்கள்.",
        "Kannada": "ಚಿಕ್ಕ ರೆಕಾರ್ಡಿಂಗ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ Voice Guard ಪತ್ತೆಮಾಡುವ ಧ್ವನಿಶಾಸ್ತ್ರೀಯ ಸೂಚನೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ. ಅಂಕ ಮತ್ತು ಮಿತಿಗಳನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ವಿವರಿಸಿದ ಎಚ್ಚರಿಕೆಯ ಫಲಿತಾಂಶ ನಿಮಗೆ ಸಿಗುತ್ತದೆ.",
    },
    "chip_formats": {
        "English": "5 audio formats",
        "Hindi": "5 ऑडियो फ़ॉर्मैट",
        "Tamil": "5 ஆடியோ வடிவங்கள்",
        "Kannada": "5 ಆಡಿಯೊ ಸ್ವರೂಪಗಳು",
    },
    "chip_languages": {
        "English": "16-language research model",
        "Hindi": "16-भाषा रिसर्च मॉडल",
        "Tamil": "16 மொழி ஆராய்ச்சி மாதிரி",
        "Kannada": "16 ಭಾಷೆಗಳ ಸಂಶೋಧನಾ ಮಾದರಿ",
    },
    "chip_cautious": {
        "English": "Three-way cautious result",
        "Hindi": "तीन-स्तरीय सावधानीपूर्ण नतीजा",
        "Tamil": "மூன்று நிலை எச்சரிக்கை முடிவு",
        "Kannada": "ಮೂರು ಹಂತದ ಎಚ್ಚರಿಕೆಯ ಫಲಿತಾಂಶ",
    },
    "workspace_kicker": {
        "English": "Analysis workspace",
        "Hindi": "विश्लेषण वर्कस्पेस",
        "Tamil": "பகுப்பாய்வு பணிப்பகுதி",
        "Kannada": "ವಿಶ್ಲೇಷಣಾ ಕಾರ್ಯಕ್ಷೇತ್ರ",
    },
    "process_title": {
        "English": "How the review works",
        "Hindi": "समीक्षा कैसे काम करती है",
        "Tamil": "ஆய்வு எப்படி செயல்படுகிறது",
        "Kannada": "ಪರಿಶೀಲನೆ ಹೇಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ",
    },
    "process_intro": {
        "English": "The interface keeps input quality, model analysis, and interpretation separate.",
        "Hindi": "इंटरफ़ेस इनपुट गुणवत्ता, मॉडल विश्लेषण और व्याख्या को अलग रखता है।",
        "Tamil": "இந்த இடைமுகம் உள்ளீட்டுத் தரம், மாதிரி பகுப்பாய்வு மற்றும் விளக்கத்தை தனித்தனியாக வைத்திருக்கிறது.",
        "Kannada": "ಇಂಟರ್ಫೇಸ್ ಇನ್‌ಪುಟ್ ಗುಣಮಟ್ಟ, ಮಾದರಿ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ವ್ಯಾಖ್ಯಾನವನ್ನು ಪ್ರತ್ಯೇಕವಾಗಿರಿಸುತ್ತದೆ.",
    },
    "step_one_title": {
        "English": "Validate the recording",
        "Hindi": "रिकॉर्डिंग की जांच",
        "Tamil": "பதிவைச் சரிபார்த்தல்",
        "Kannada": "ರೆಕಾರ್ಡಿಂಗ್ ಪರಿಶೀಲನೆ",
    },
    "step_one_body": {
        "English": "Format, duration, level, and active speech are checked before loading the model.",
        "Hindi": "मॉडल लोड होने से पहले फ़ॉर्मैट, अवधि, ध्वनि स्तर और सक्रिय भाषण जांचे जाते हैं।",
        "Tamil": "மாதிரி ஏற்றப்படுவதற்கு முன் வடிவம், நீளம், ஒலி அளவு மற்றும் செயலில் உள்ள பேச்சு சரிபார்க்கப்படும்.",
        "Kannada": "ಮಾದರಿಯನ್ನು ಲೋಡ್ ಮಾಡುವ ಮೊದಲು ಸ್ವರೂಪ, ಅವಧಿ, ಧ್ವನಿ ಮಟ್ಟ ಮತ್ತು ಸಕ್ರಿಯ ಮಾತನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತದೆ.",
    },
    "step_two_title": {
        "English": "Inspect acoustic patterns",
        "Hindi": "ध्वनिक पैटर्न का विश्लेषण",
        "Tamil": "ஒலியியல் வடிவங்களை ஆய்வு செய்தல்",
        "Kannada": "ಧ್ವನಿಶಾಸ್ತ್ರೀಯ ಮಾದರಿಗಳ ಪರಿಶೀಲನೆ",
    },
    "step_two_body": {
        "English": "A frozen multilingual speech encoder and classifier score synthetic-speech characteristics.",
        "Hindi": "बहुभाषी स्पीच एन्कोडर और क्लासिफ़ायर सिंथेटिक आवाज़ की विशेषताओं को स्कोर करते हैं।",
        "Tamil": "பலமொழி பேச்சு encoder மற்றும் classifier செயற்கைப் பேச்சின் பண்புகளை மதிப்பிடுகின்றன.",
        "Kannada": "ಬಹುಭಾಷಾ ಸ್ಪೀಚ್ ಎನ್‌ಕೋಡರ್ ಮತ್ತು ಕ್ಲಾಸಿಫೈಯರ್ ಕೃತಕ ಮಾತಿನ ಲಕ್ಷಣಗಳಿಗೆ ಅಂಕ ನೀಡುತ್ತವೆ.",
    },
    "step_three_title": {
        "English": "Present a cautious result",
        "Hindi": "सावधानीपूर्ण नतीजा",
        "Tamil": "எச்சரிக்கையான முடிவு",
        "Kannada": "ಎಚ್ಚರಿಕೆಯ ಫಲಿತಾಂಶ",
    },
    "step_three_body": {
        "English": "The score maps to likely genuine, inconclusive, or likely synthetic—never proof.",
        "Hindi": "स्कोर से संभवतः असली, अनिर्णायक या संभवतः सिंथेटिक नतीजा मिलता है—यह प्रमाण नहीं है।",
        "Tamil": "மதிப்பெண் உண்மையாக இருக்கலாம், முடிவற்றது அல்லது செயற்கையாக இருக்கலாம் எனக் காட்டும்—இது சான்று அல்ல.",
        "Kannada": "ಅಂಕವು ಬಹುಶಃ ನೈಜ, ಅನಿಶ್ಚಿತ ಅಥವಾ ಬಹುಶಃ ಕೃತಕ ಎಂದು ಸೂಚಿಸುತ್ತದೆ—ಇದು ಪುರಾವೆಯಲ್ಲ.",
    },
    "session_note": {
        "English": "Your clip is processed inside the current app session. Avoid confidential audio on a public demo URL.",
        "Hindi": "आपकी क्लिप मौजूदा ऐप सत्र में प्रोसेस होती है। सार्वजनिक डेमो URL पर गोपनीय ऑडियो अपलोड न करें।",
        "Tamil": "உங்கள் கிளிப் தற்போதைய பயன்பாட்டு அமர்வில் செயலாக்கப்படுகிறது. பொது demo URL-இல் ரகசிய ஆடியோவைப் பதிவேற்ற வேண்டாம்.",
        "Kannada": "ನಿಮ್ಮ ಕ್ಲಿಪ್ ಪ್ರಸ್ತುತ ಆ್ಯಪ್ ಸೆಷನ್‌ನಲ್ಲಿ ಪ್ರಕ್ರಿಯೆಯಾಗುತ್ತದೆ. ಸಾರ್ವಜನಿಕ ಡೆಮೊ URL‌ಗೆ ಗೌಪ್ಯ ಆಡಿಯೊ ಅಪ್‌ಲೋಡ್ ಮಾಡಬೇಡಿ.",
    },
    "result_label": {
        "English": "Analysis result",
        "Hindi": "विश्लेषण नतीजा",
        "Tamil": "பகுப்பாய்வு முடிவு",
        "Kannada": "ವಿಶ್ಲೇಷಣಾ ಫಲಿತಾಂಶ",
    },
    "score_ring_label": {
        "English": "AI signal",
        "Hindi": "एआई संकेत",
        "Tamil": "AI சுட்டி",
        "Kannada": "AI ಸೂಚನೆ",
    },
    "interpretation_title": {
        "English": "Interpret with care",
        "Hindi": "सावधानी से समझें",
        "Tamil": "கவனமாக விளக்குங்கள்",
        "Kannada": "ಎಚ್ಚರಿಕೆಯಿಂದ ಅರ್ಥೈಸಿ",
    },
    "transparency_title": {
        "English": "Model transparency",
        "Hindi": "मॉडल पारदर्शिता",
        "Tamil": "மாதிரி வெளிப்படைத்தன்மை",
        "Kannada": "ಮಾದರಿ ಪಾರದರ್ಶಕತೆ",
    },
    "transparency_body": {
        "English": "Noise, compression, replay, short speech, and unseen generators can materially change performance.",
        "Hindi": "शोर, कंप्रेशन, दोबारा चलाया गया ऑडियो, छोटी स्पीच और नए जनरेटर प्रदर्शन को काफ़ी बदल सकते हैं।",
        "Tamil": "சத்தம், சுருக்கம், மீள்பதிவு, குறுகிய பேச்சு மற்றும் காணாத உருவாக்கிகள் செயல்திறனை குறிப்பிடத்தக்க அளவில் மாற்றலாம்.",
        "Kannada": "ಶಬ್ದ, ಕಂಪ್ರೆಷನ್, ಮರುಪ್ಲೇ, ಚಿಕ್ಕ ಮಾತು ಮತ್ತು ಕಾಣದ ಜನರೇಟರ್‌ಗಳು ಕಾರ್ಯಕ್ಷಮತೆಯನ್ನು ಗಮನಾರ್ಹವಾಗಿ ಬದಲಿಸಬಹುದು.",
    },
    "footer_product": {
        "English": "Voice Guard · Indic synthetic-speech research demo",
        "Hindi": "Voice Guard · भारतीय सिंथेटिक-स्पीच रिसर्च डेमो",
        "Tamil": "Voice Guard · இந்திய செயற்கைப் பேச்சு ஆராய்ச்சி டெமோ",
        "Kannada": "Voice Guard · ಭಾರತೀಯ ಕೃತಕ-ಮಾತು ಸಂಶೋಧನಾ ಡೆಮೊ",
    },
})

# Rename the report throughout the interface: it is an automated analysis, not
# forensic evidence.
TRANSLATIONS["pdf_heading"] = {
    "English": "Automated Analysis Report", "Hindi": "स्वचालित विश्लेषण रिपोर्ट",
    "Tamil": "தானியங்கி பகுப்பாய்வு அறிக்கை", "Kannada": "ಸ್ವಯಂಚಾಲಿತ ವಿಶ್ಲೇಷಣಾ ವರದಿ",
}
TRANSLATIONS["pdf_button"] = {
    "English": "Download analysis report", "Hindi": "विश्लेषण रिपोर्ट डाउनलोड करें",
    "Tamil": "பகுப்பாய்வு அறிக்கையைப் பதிவிறக்கவும்", "Kannada": "ವಿಶ್ಲೇಷಣಾ ವರದಿಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
}
TRANSLATIONS["pdf_body"] = {
    "English": "Download a shareable record of this automated result, its input hash, and its limitations.",
    "Hindi": "इस स्वचालित नतीजे, इनपुट हैश और इसकी सीमाओं का साझा रिकॉर्ड डाउनलोड करें।",
    "Tamil": "இந்த தானியங்கி முடிவு, உள்ளீட்டு hash மற்றும் அதன் வரம்புகளின் பகிரக்கூடிய பதிவைப் பதிவிறக்கவும்.",
    "Kannada": "ಈ ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ, ಇನ್‌ಪುಟ್ hash ಮತ್ತು ಅದರ ಮಿತಿಗಳ ಹಂಚಬಹುದಾದ ದಾಖಲೆಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ.",
}

# Transmission robustness comparison added to the upload workflow. Technical
# condition names (Opus/G.711/8 kHz) remain standards-based proper names.
TRANSLATIONS.update({
    "robustness_title": {
        "English": "Transmission robustness check",
        "Hindi": "प्रसारण मजबूती जाँच",
        "Tamil": "பரிமாற்ற நிலைத்தன்மைச் சோதனை",
        "Kannada": "ಪ್ರಸರಣ ಸ್ಥಿರತೆ ಪರಿಶೀಲನೆ",
    },
    "robustness_body": {
        "English": "Compare this result after common messaging and telephone transformations.",
        "Hindi": "सामान्य मैसेजिंग और टेलीफोन बदलावों के बाद इस परिणाम की तुलना करें।",
        "Tamil": "பொதுவான செய்தியிடல் மற்றும் தொலைபேசி மாற்றங்களுக்குப் பிறகு முடிவை ஒப்பிடுங்கள்.",
        "Kannada": "ಸಾಮಾನ್ಯ ಮೆಸೇಜಿಂಗ್ ಮತ್ತು ದೂರವಾಣಿ ಬದಲಾವಣೆಗಳ ನಂತರ ಫಲಿತಾಂಶವನ್ನು ಹೋಲಿಸಿ.",
    },
    "robustness_button": {
        "English": "Check transmission robustness",
        "Hindi": "प्रसारण मजबूती जाँचें",
        "Tamil": "பரிமாற்ற நிலைத்தன்மையைச் சோதிக்கவும்",
        "Kannada": "ಪ್ರಸರಣ ಸ್ಥಿರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ",
    },
    "robustness_running": {
        "English": "Testing common transmission conditions…",
        "Hindi": "सामान्य प्रसारण स्थितियों की जाँच हो रही है…",
        "Tamil": "பொதுவான பரிமாற்ற நிலைகள் சோதிக்கப்படுகின்றன…",
        "Kannada": "ಸಾಮಾನ್ಯ ಪ್ರಸರಣ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಪರೀಕ್ಷಿಸಲಾಗುತ್ತಿದೆ…",
    },
    "robustness_condition": {
        "English": "Condition", "Hindi": "स्थिति",
        "Tamil": "நிலை", "Kannada": "ಪರಿಸ್ಥಿತಿ",
    },
    "robustness_score": {
        "English": "AI-likelihood", "Hindi": "एआई-संभावना",
        "Tamil": "AI சாத்தியம்", "Kannada": "AI ಸಾಧ್ಯತೆ",
    },
    "robustness_change": {
        "English": "Change", "Hindi": "बदलाव",
        "Tamil": "மாற்றம்", "Kannada": "ಬದಲಾವಣೆ",
    },
    "robustness_verdict": {
        "English": "Result", "Hindi": "परिणाम",
        "Tamil": "முடிவு", "Kannada": "ಫಲಿತಾಂಶ",
    },
    "robustness_stability": {
        "English": "Stability", "Hindi": "स्थिरता",
        "Tamil": "நிலைத்தன்மை", "Kannada": "ಸ್ಥಿರತೆ",
    },
    "robustness_unchanged": {
        "English": "Unchanged", "Hindi": "अपरिवर्तित",
        "Tamil": "மாறவில்லை", "Kannada": "ಬದಲಾಗಿಲ್ಲ",
    },
    "robustness_changed": {
        "English": "Changed", "Hindi": "बदला",
        "Tamil": "மாறியது", "Kannada": "ಬದಲಾಗಿದೆ",
    },
    "robustness_failed": {
        "English": "Unavailable", "Hindi": "अनुपलब्ध",
        "Tamil": "கிடைக்கவில்லை", "Kannada": "ಲಭ್ಯವಿಲ್ಲ",
    },
    "robustness_summary_stable": {
        "English": "The verdict stayed the same in every successful condition.",
        "Hindi": "हर सफल स्थिति में परिणाम समान रहा।",
        "Tamil": "வெற்றிகரமான எல்லா நிலைகளிலும் முடிவு மாறவில்லை.",
        "Kannada": "ಯಶಸ್ವಿಯಾದ ಪ್ರತಿಯೊಂದು ಪರಿಸ್ಥಿತಿಯಲ್ಲೂ ಫಲಿತಾಂಶ ಬದಲಾಗಲಿಲ್ಲ.",
    },
    "robustness_no_comparisons": {
        "English": "No transformed condition completed successfully.",
        "Hindi": "कोई भी बदली हुई स्थिति सफलतापूर्वक पूरी नहीं हुई।",
        "Tamil": "மாற்றப்பட்ட எந்த நிலையும் வெற்றிகரமாக நிறைவடையவில்லை.",
        "Kannada": "ಯಾವುದೇ ಬದಲಾಯಿಸಿದ ಪರಿಸ್ಥಿತಿ ಯಶಸ್ವಿಯಾಗಿ ಪೂರ್ಣಗೊಂಡಿಲ್ಲ.",
    },
    "robustness_summary_changed": {
        "English": "The verdict changed in {count} of {total} successful transformed conditions.",
        "Hindi": "{total} सफल बदली हुई स्थितियों में से {count} में परिणाम बदल गया।",
        "Tamil": "{total} வெற்றிகரமான மாற்றப்பட்ட நிலைகளில் {count} இல் முடிவு மாறியது.",
        "Kannada": "{total} ಯಶಸ್ವಿ ಬದಲಾಯಿಸಿದ ಪರಿಸ್ಥಿತಿಗಳಲ್ಲಿ {count}ರಲ್ಲಿ ಫಲಿತಾಂಶ ಬದಲಾಗಿದೆ.",
    },
    "robustness_max_change": {
        "English": "Largest AI-likelihood change from the original: {value:.1f} percentage points.",
        "Hindi": "मूल ऑडियो से एआई-संभावना में सबसे बड़ा बदलाव: {value:.1f} प्रतिशत अंक।",
        "Tamil": "அசலிலிருந்து AI சாத்தியத்தின் மிகப்பெரிய மாற்றம்: {value:.1f} சதவீதப் புள்ளிகள்.",
        "Kannada": "ಮೂಲದಿಂದ AI ಸಾಧ್ಯತೆಯ ಅತಿದೊಡ್ಡ ಬದಲಾವಣೆ: {value:.1f} ಶೇಕಡಾವಾರು ಅಂಕಗಳು.",
    },
    "robustness_partial_failure": {
        "English": "Some conditions could not be tested: {conditions}.",
        "Hindi": "कुछ स्थितियों की जाँच नहीं हो सकी: {conditions}।",
        "Tamil": "சில நிலைகளைச் சோதிக்க முடியவில்லை: {conditions}.",
        "Kannada": "ಕೆಲವು ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಪರೀಕ್ಷಿಸಲಾಗಲಿಲ್ಲ: {conditions}.",
    },
    "robustness_limit": {
        "English": "This compares the current model under simulated channels; it does not prove that the model is codec-robust.",
        "Hindi": "यह मौजूदा मॉडल की कृत्रिम चैनलों में तुलना है; यह मॉडल के कोडेक-मजबूत होने का प्रमाण नहीं है।",
        "Tamil": "இது தற்போதைய மாதிரியை உருவகப்படுத்திய சேனல்களில் ஒப்பிடுகிறது; மாதிரி codec-நிலைத்தன்மை கொண்டது என்பதை இது நிரூபிக்காது.",
        "Kannada": "ಇದು ಪ್ರಸ್ತುತ ಮಾದರಿಯನ್ನು ಅನುಕರಿಸಿದ ಚಾನೆಲ್‌ಗಳಲ್ಲಿ ಹೋಲಿಸುತ್ತದೆ; ಮಾದರಿ codec-ಸ್ಥಿರವಾಗಿದೆ ಎಂದು ಇದು ಸಾಬೀತುಪಡಿಸುವುದಿಲ್ಲ.",
    },
})


def t(key: str, lang: str) -> str:
    """Look up a translated UI string. Falls back to English, then the raw key,
    so a missing translation degrades to readable text instead of a crash."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("English") or key
