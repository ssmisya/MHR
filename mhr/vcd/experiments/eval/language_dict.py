language_dict = {
    'en':{'full_name':'English','yes':'yes','no':'no','prompt_suffix':' Please select your answer from ({},{}). And make sure that your answer should not contain any other word than given answers.',"example":["apple","Where is the park?"]},
    'zh':{'full_name':'Chinese','yes':'是','no':['否','不','不是'],'prompt_suffix':' 请从以下答案中选择您的答案:({},{})。并确保您的答案不包含任何其他单词。',"example":["苹果","公园在哪里？"]},
    'ja':{'full_name':'Japanese','yes':'はい','no':'いいえ','prompt_suffix':' 以下の答えからお選びください:({},{})。そして、あなたの答えには他の言葉が含まれていないことを確認してください。',"example":["りんご","公園はどこですか？"]},
    'ko':{'full_name':'Korean','yes':'예','no':'아니요','prompt_suffix':' 다음 답변 중에서 선택하십시오:({},{})。그리고 당신의 답변에는 다른 단어가 포함되지 않도록 주의하십시오.',"example":["사과","공원은 어디에 있습니까?"]},
    'es':{'full_name':'Spanish','yes':'sí','no':'no','prompt_suffix':' Por favor, seleccione su respuesta de entre ({},{}). Y asegúrese de que su respuesta no contenga ninguna otra palabra.',"example":["manzana","¿Dónde está el parque?"]},
    'fr':{'full_name':'French','yes':'oui','no':'non','prompt_suffix':' Veuillez sélectionner votre réponse parmi ({},{}). Et assurez-vous que votre réponse ne contient aucun autre mot.',"example":["pomme","Où est le parc?"]},
    'de':{'full_name':'German','yes':'ja','no':'nein','prompt_suffix':' Bitte wählen Sie Ihre Antwort aus ({},{}). Und stellen Sie sicher, dass Ihre Antwort kein anderes Wort enthält.',"example":["Apfel","Wo ist der Park?"]},
    'it':{'full_name':'Italian','yes':'sì','no':'no','prompt_suffix':' Si prega di selezionare la risposta da ({},{}). E assicurarsi che la risposta non contenga altre parole.',"example":["mela","Dov'è il parco?"]},
    'pt':{'full_name':'Portuguese','yes':'sim','no':'não','prompt_suffix':' Por favor, selecione sua resposta de ({},{}). E certifique-se de que sua resposta não contenha nenhuma outra palavra.',"example":["maçã","Onde está o parque?"]},
    'ru':{'full_name':'Russian','yes':'да','no':'нет','prompt_suffix':' Пожалуйста, выберите ваш ответ из ({},{}). И убедитесь, что ваш ответ не содержит никаких других слов.',"example":["яблоко","Где парк?"]},
    'ar':{'full_name':'Arabic','yes':'نعم','no':'لا','prompt_suffix':' يرجى اختيار إجابتك من ({},{}). وتأكد من أن إجابتك لا تحتوي على أي كلمة أخرى.',"example":["تفاحة","أين الحديقة؟"]},
    'tr':{'full_name':'Turkish','yes':'evet','no':'hayır','prompt_suffix':' Lütfen cevabınızı ({},{}). seçin. Ve cevabınızın başka bir kelime içermediğinden emin olun.',"example":["elma","Park nerede?"]},
    'vi':{'full_name':'Vietnamese','yes':['có','Có','Đúng'],'no':'không','prompt_suffix':' Vui lòng chọn câu trả lời của bạn từ ({},{}). Và đảm bảo rằng câu trả lời của bạn không chứa bất kỳ từ nào khác.',"example":["táo","Công viên ở đâu?"]},
    'th':{'full_name':'Thai','yes':'ใช่','no':'ไม่','prompt_suffix':' โปรดเลือกคำตอบของคุณจาก ({},{}). และตรวจสอบให้แน่ใจว่าคำตอบของคุณไม่มีคำอื่นๆ.',"example":["แอปเปิ้ล","สวนอยู่ที่ไหน?"]},
    'id':{'full_name':'Indonesian','yes':'ya','no':'tidak','prompt_suffix':' Silakan pilih jawaban Anda dari ({},{}). Dan pastikan bahwa jawaban Anda tidak mengandung kata lain.',"example":["apel","Di mana taman?"]},
    'ms':{'full_name':'Malay','yes':'ya','no':'tidak','prompt_suffix':' Sila pilih jawapan anda dari ({},{}). Dan pastikan bahawa jawapan anda tidak mengandungi sebarang perkataan lain.',"example":["epal","Di mana taman?"]},
    'nl':{'full_name':'Dutch','yes':'ja','no':'nee','prompt_suffix':' Selecteer uw antwoord uit ({},{}). En zorg ervoor dat uw antwoord geen andere woorden bevat.',"example":["appel","Waar is het park?"]},
    'sv':{'full_name':'Swedish','yes':'ja','no':'nej','prompt_suffix':' Vänligen välj ditt svar från ({},{}). Och se till att ditt svar inte innehåller några andra ord.',"example":["äpple","Var är parken?"]},
    'fa':{'full_name':'Persian','yes':['بله','ہے','ہے','است','جی.','نعم،'],'no':'نه','prompt_suffix':' لطفاً پاسخ خود را از ({},{}). انتخاب کنید و مطمئن شوید که پاسخ شما شامل هیچ کلمه دیگری نیست.',"example":["سیب","پارک کجاست؟"]},
    'el':{'full_name':'Greek','yes':['ναι','Ναι'],'no':['όχι','οχι','Όχι'],'prompt_suffix':' Παρακαλώ επιλέξτε την απάντησή σας από ({},{}). Και βεβαιωθείτε ότι η απάντησή σας δεν περιέχει καμία άλλη λέξη.',"example":["μήλο","Πού είναι το πάρκο;"]},
    'uk':{'full_name':'Ukrainian','yes':'так','no':['ні','нет'],'prompt_suffix':' Будь ласка, виберіть свою відповідь з ({},{}). І переконайтеся, що ваша відповідь не містить жодного іншого слова.',"example":["яблуко","Де парк?"]},
    'bg':{'full_name':'Bulgarian','yes':'да','no':'не','prompt_suffix':' Моля, изберете отговора си от ({},{}). И уверете се, че отговорът ви не съдържа никакви други думи.',"example":["ябълка","Къде е паркът?"]},
    'hi':{'full_name':'Hindi','yes':'हाँ','no':'नहीं','prompt_suffix':' कृपया अपना उत्तर चुनें ({},{}). और सुनिश्चित करें कि आपका उत्तर दिए गए उत्तरों के अलावा कोई अन्य शब्द नहीं है।',"example":["सेब","पार्क कहाँ है?"]},
    'ta':{'full_name':'Tamil','yes':'ஆம்','no':'இல்லை','prompt_suffix':' உங்கள் பதிலை தேர்ந்தெடுக்கவும் ({},{}). மற்ற வார்த்தைகள் இல்லாமல் உங்கள் பதில் உள்ளதாக உறுதிப்படுத்தவும்.',"example":["ஆப்பிள்","பூங்கா எங்கே உள்ளது?"]},
    'bn':{'full_name':'Bengali','yes':'হ্যাঁ','no':'না','prompt_suffix':' দয়া করে আপনার উত্তর নির্বাচন করুন ({},{}). এবং নিশ্চিত করুন যে আপনার উত্তরে অন্য কোনও শব্দ নেই।',"example":["আপেল","পার্ক কোথায়?"]},
    'ur':{'full_name':'Urdu','yes':'جی ہاں','no':'نہیں','prompt_suffix':' براہ کرم اپنے جواب کو منتخب کریں ({},{}). اور یہ یقینی بنائیں کہ آپ کا جواب دیے گئے جوابات کے علاوہ کوئی دوسرا لفظ نہیں ہے۔',"example":["سیب","پارک کہاں ہے؟"]},
    'ml':{'full_name':'Malayalam','yes':'അതെ','no':'ഇല്ല','prompt_suffix':' ദയവായി നിങ്ങളുടെ ഉത്തരം തിരഞ്ഞെടുക്കുക ({},{}). മറ്റ് വാക്കുകളൊന്നും നിങ്ങളുടെ ഉത്തരത്തിൽ ഉള്ളതാകാതിരിക്കാം.',"example":["ആപ്പിൾ","പാര്‍ക്ക് എവിടെയാണ്?"]},
    'mr':{'full_name':'Marathi','yes':'हो','no':'नाही','prompt_suffix':' कृपया आपले उत्तर निवडा ({},{}). आणि खात्री करा की आपले उत्तरामध्ये दिलेल्या उत्तरांपेक्षा इतर कोणतेही शब्द नसतील.',"example":["सफरचंद","उद्यान कुठे आहे?"]},
    'te':{'full_name':'Telugu','yes':'అవును','no':'కాదు','prompt_suffix':' దయచేసి మీ సమాధానాన్ని ఎంచుకోండి ({},{}). మరియు మీ సమాధానంలో ఇతర ఏ పదం లేదు అని ఖచ్చితంగా చూసుకోండి.',"example":["ఆపిల్","పార్కు ఎక్కడ ఉంది?"]},
    'gu':{'full_name':'Gujarati','yes':'હા','no':'નહીં','prompt_suffix':' કૃપા કરીને તમારો જવાબ પસંદ કરો ({},{}). અને ખાતરી કરો કે તમારો જવાબ આપેલા જવાબો કેમ કોઈ અન્ય શબ્દ નથી.',"example":["સફરજંદ","પાર્ક ક્યાં છે?"]},
    'my':{'full_name':'Burmese','yes':'ဟု','no':'မဟု','prompt_suffix':' ကျေးဇူးပြု၍ သင့်အဖြစ်အစားသင့်အဖြစ်ကို ရွေးချယ်ပါ ({},{}။)။ နှင့် သင့်အဖြစ်သင့်အဖြစ်ကို အခြားစာလုံးမရှိပါက သင့်အဖြစ်ကို သတ်မှတ်ပါ။',"example":["ပန်းသီး","ပတ်ဝန်မကျောင်းဖြစ်သည်။"]},
    'jv':{'full_name':'Javanese','yes':['ya','iya'],'no':'ora','prompt_suffix':' Mugi pilih jawaban sampeyan saka ({},{}). Lan pastikan yèn jawaban sampeyan ora duwe pitakon liyane.',"example":["apel","Taman neng ndi?"]},
    'sw':{'full_name':'Swahili','yes':'ndiyo','no':'hapana','prompt_suffix':' Tafadhali chagua jibu lako kutoka kwa ({},{}). Na hakikisha kwamba jibu lako halina neno lingine lolote.',"example":["tufaha","Hifadhi iko wapi?"]},
}

nllb_200_distilled_600M_language_dict = {
        'Swahili' :'swh_Latn',
        'Chinese' : "zho_Hans",
        "Bengali" : "ben_Beng",
        "German" : "deu_Latn",
        "Spanish" : "spa_Latn",
        "French" : "fra_Latn",
        "Japanese" : "jpn_Jpan",
        "Russian" : "rus_Cyrl",
        "Thai" : "tha_Thai",
        "English" : "eng_Latn",
        "Arabic" : "arb_Arab",
        "Portuguese" : "por_Latn",
        "Turkish" : "tur_Latn",
        "Vietnamese" : "vie_Latn",
        "Italian" : "ita_Latn",
        "Korean" : "kor_Hang",
        "Dutch" : "nld_Latn",
        "Polish" : "pol_Latn",
        "Greek" : "ell_Grek",
        "Hindi" : "hin_Deva",
        "Indonesian" : "ind_Latn",
        "Czech" : "ces_Latn",
        "Swedish" : "swe_Latn",
        "Marathi" : "mar_Deva",
        "Ukrainian" : "ukr_Cyrl",
        "Bulgarian" : "bul_Cyrl",
        "Finnish" : "fin_Latn",
        "Urdu" : "urd_Arab",
        'Vietnamese' : 'vie_Latn',
        "Malay" : "zsm_Latn",
        "Persian" : "pes_Arab",
        "Tamil" : "tam_Taml",
        "Malayalam" : "mal_Mlym",
        "Telugu": "tel_Telu",
        "Gujarati" : "guj_Gujr",
        "Burmese" : "mya_Mymr",
        "Javanese" : "jav_Latn",
        } 

system_conv_dict=dict(
    en = "A chat between a curious human and an artificial intelligence assistant. " 
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    zh = "一个好奇的人和人工智能助手之间的聊天。 " 
              "助手对人类的问题给出了有用、详细和礼貌的回答。",
    ja = "好奇心旺盛な人間と人工知能アシスタントの間のチャット。 " 
                "アシスタントは人間の質問に対して有益で詳細かつ礼儀正しい回答をします。",
    ko = "호기심 많은 사람과 인공 지능 어시스턴트 간의 대화. " 
                "어시스턴트는 인간의 질문에 유용하고 상세하며 정중한 답변을 제공합니다.",
    es = "Una conversación entre un humano curioso y un asistente de inteligencia artificial. " 
                "El asistente da respuestas útiles, detalladas y educadas a las preguntas del humano.",
    fr = "Une conversation entre un humain curieux et un assistant d'intelligence artificielle. " 
                "L'assistant donne des réponses utiles, détaillées et polies aux questions de l'humain.",
    de = "Ein Gespräch zwischen einem neugierigen Menschen und einem künstlichen Intelligenzassistenten. " 
                "Der Assistent gibt hilfreiche, detaillierte und höfliche Antworten auf die Fragen des Menschen.",
    it = "Una conversazione tra un umano curioso e un assistente di intelligenza artificiale. " 
                "L'assistente fornisce risposte utili, dettagliate e cortesi alle domande dell'umano.",
    pt = "Uma conversa entre um humano curioso e um assistente de inteligência artificial. " 
                "O assistente dá respostas úteis, detalhadas e educadas às perguntas do humano.",
    ru = "Чат между любопытным человеком и искусственным интеллектом-помощником. "     
                "Помощник дает полезные, подробные и вежливые ответы на вопросы человека.",
    ar = "محادثة بين إنسان فضولي ومساعد ذكاء اصطناعي. " 
                "يقدم المساعد إجابات مفيدة ومفصلة ومهذبة لأسئلة الإنسان.",
    tr = "Meraklı bir insan ve yapay zeka asistanı arasında bir sohbet. " 
                "Asistan, insanın sorularına yararlı, detaylı ve kibar cevaplar verir.",
    vi = "Cuộc trò chuyện giữa một con người tò mò và một trợ lý trí tuệ nhân tạo. " 
                "Trợ lý đưa ra câu trả lời hữu ích, chi tiết và lịch sự cho câu hỏi của con người.",
    th = "การสนทนาระหว่างมนุษย์ที่อยากรู้อยากเห็นและผู้ช่วยปัญญาประดิษฐ์ " 
                "ผู้ช่วยให้คำตอบที่เป็นประโยชน์ ละเอียดและสุภาพต่อคำถามของมนุษย์",
    id = "Percakapan antara manusia yang ingin tahu dan asisten kecerdasan buatan. " 
                "Asisten memberikan jawaban yang membantu, rinci, dan sopan untuk pertanyaan manusia.",
    ms = "Perbualan antara manusia yang ingin tahu dan pembantu kecerdasan buatan. " 
                "Pembantu memberikan jawapan yang membantu, terperinci, dan sopan kepada soalan manusia.",
    nl = "Een gesprek tussen een nieuwsgierig mens en een kunstmatige intelligentie-assistent. " 
                "De assistent geeft behulpzame, gedetailleerde en beleefde antwoorden op de vragen van de mens.",
    sv = "En konversation mellan en nyfiken människa och en artificiell intelligensassistent. " 
                "Assistenten ger hjälpsamma, detaljerade och artiga svar på människans frågor.",
    fa = "یک گفتگو بین یک انسان کنجکاو و یک دستیار هوش مصنوعی. " 
                "دستیار پاسخ‌های مفید، دقیق و مودب به سوالات انسان را ارائه می‌دهد.",
    uk = "Розмова між цікавою людиною та штучним інтелектуальним помічником. " 
                "Помічник дає корисні, детальні та ввічливі відповіді на питання людини.",
    bg = "Разговор между любопитен човек и изкуствен интелигентен асистент. " 
                "Асистентът дава полезни, подробни и учтиви отговори на въпросите на човека.",
    hi = "एक उत्सुक मानव और एक कृत्रिम बुद्धिमत्ता सहायक के बीच एक चैट। " 
                "सहायक मानव के प्रश्नों के लिए उपयोगी, विस्तृत और शिष्ट उत्तर देता है।",
    ta = "ஒரு குறிப்பாளர் மற்றும் ஒரு கருத்துரை உதவி இடையே ஒரு உரை. " 
                "உதவி மனிதரின் கேள்விகளுக்கு உபயோகமான, விரிவான மற்றும் பொதுவான பதில்களை வழங்குகிறான்.",
)

def compare_str_list(str1,str2):
    if isinstance(str1,list):
        for i in str1:
            if i in str2:
                return True
        return False
    elif isinstance(str1,str):
        return str1 in str2
    else:
        raise ValueError("str1 should be list or str")
    
translate_instruction="""
 You are a helpful AI assistant translator. You are given a piece of sentence or a word, and you are asked to translate it from {} into {}. Be aware that your answer should only contain the translation of the sentence or word. Make sure your answer is accurate and does not contain any other word.
"""

def get_translation_prompt(target_language,source_language="en"):
    src_full = language_dict[source_language]['full_name']
    tgt_full = language_dict[target_language]['full_name']
    
    prompt = translate_instruction.format(src_full,tgt_full)
    few_shot = [(language_dict[source_language]['example'][0],language_dict[target_language]['example'][0]),(language_dict[source_language]['example'][1],language_dict[target_language]['example'][1])]
    messages=[{"role": "system", "content":prompt}]
    for shot in few_shot:
        messages.append({"role": "user", "content": shot[0]})
        messages.append({"role": "assistant", "content": shot[1]})
    return messages

def llava_v1_get_language_conv(lang):
    from mhr.alignment.models.llava_v1_5.llava.conversation import SeparatorStyle,Conversation
    # system_conv = system_conv_dict[lang]
    system_conv = f"A chat between a curious human and an artificial intelligence assistant. The chat is in {language_dict[lang]['full_name']}." + \
                    f"The assistant gives helpful, detailed, and polite answers in {language_dict[lang]['full_name']}"
    conv = Conversation(
        system=system_conv,
        roles=("USER", "ASSISTANT"),
        version="v1",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    return conv