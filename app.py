from flask import Flask, render_template, redirect, request
import joblib
app = Flask(__name__)

rfc = joblib.load('rfc.pkl')
test1 = {'Drug Reaction': 'Blood test', 'Malaria': 'Blood smear test', 'Allergy': 'Prick skin testing', 'Hypothyroidism': 'TSH test', 'Psoriasis': 'skin biopsy', 'GERD': 'Upper endoscopy', 'Chronic cholestasis': 'CT scan', 'hepatitis A': 'Hepatitis A IgM', 'Osteoarthristis': 'Blood tests', '(vertigo) Paroymsal  Positional Vertigo': 'Electronystagmography (ENG)', 'Hypoglycemia': 'Blood tests', 'Acne': 'HbA1C test', 'Diabetes': 'Blood sugar test', 'Impetigo': 'Lab test generally are not needed', 'Hypertension': 'Ambulatory monitoring', 'Peptic ulcer diseae': 'Urea breath test', 'Dimorphic hemmorhoids(piles)': 'Lab tests are not needed', 'Common cold': 'Lab tests are not needed', 'Chicken pox': 'Blood tests', 'Cervical spondylosis': 'Neck X-ray', 'Hyperthyroidism': 'Blood tests',
         'Urinary tract infection': 'Urinalysis', 'Varicose veins': 'Venous Doppler ultrasound', 'AIDS': 'Antigen test', 'Paralysis (brain hemorrhage)': 'CT scan or MRI', 'Typhoid': 'Typhidot test', 'Hepatitis B': 'HBsAg test', 'Fungal infection': 'Mycology blood test', 'Hepatitis C': 'HCV antibody test', 'Migraine': 'MRI or CT scan', 'Bronchial Asthma': 'Spirometry', 'Alcoholic hepatitis': 'Serum bilirubin test and ALT test', 'Jaundice': 'urinalysis and HIDA scan', 'Hepatitis E': 'Anti HEV IgM test', 'Dengue': 'Dengue NS1 Antigen test', 'Hepatitis D': 'Liver function test', 'Heart Attack': 'Echocardiogram test', 'Pneumonia': 'Sputum test', 'Arthritis': 'ESR and CRP test', 'Gastroenteritis': 'Renal function test', 'Tuberculosis': 'TB blood test'}

unique = ['itching', 'skin rash', 'nodal skin eruptions', 'dischromic  patches', 'continuous sneezing', 'shivering', 'chills', 'watering from eyes', 'stomach pain', 'acidity', 'ulcers on tongue', 'vomiting', 'cough', 'chest pain', 'yellowish skin', 'nausea', 'loss of appetite', 'abdominal pain', 'yellowing of eyes', 'burning micturition', 'spotting  urination', 'passage of gases', 'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'high fever', 'extra marital contacts', 'fatigue', 'weight loss', 'restlessness', 'lethargy', 'irregular sugar level', 'blurred and distorted vision', 'obesity', 'excessive hunger', 'increased appetite', 'polyuria', 'sunken eyes', 'dehydration', 'diarrhoea', 'breathlessness', 'family history', 'mucoid sputum', 'headache', 'dizziness', 'loss of balance', 'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances', 'back pain', 'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium', 'dark urine', 'sweating', 'muscle pain', 'mild fever', 'swelled lymph nodes', 'malaise', 'red spots over body', 'joint pain', 'pain behind the eyes', 'constipation',
          'toxic look (typhos)', 'belly pain', 'yellow urine', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'acute liver failure', 'swelling of stomach', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'phlegm', 'blood in sputum', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'loss of smell', 'fast heart rate', 'rusty sputum', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'cramps', 'bruising', 'swollen legs', 'swollen blood vessels', 'prominent veins on calf', 'weight gain', 'cold hands and feets', 'mood swings', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'abnormal menstruation', 'muscle weakness', 'anxiety', 'slurred speech', 'palpitations', 'drying and tingling lips', 'knee pain', 'hip joint pain', 'swelling joints', 'painful walking', 'movement stiffness', 'spinning movements', 'unsteadiness', 'pus filled pimples', 'blackheads', 'scurring', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze']
desc = {'Drug Reaction': 'An adverse drug reaction (ADR) is an injury caused by taking medication. ADRs may occur following a single dose or prolonged administration of a drug or result from the combination of two or more drugs.', 'Malaria': 'An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito or by a contaminated needle or transfusion. Falciparum malaria is the most deadly type.', 'Allergy': "An allergy is an immune system response to a foreign substance that's not typically harmful to your body.They can include certain foods, pollen, or pet dander. Your immune system's job is to keep you healthy by fighting harmful pathogens.", 'Hypothyroidism': 'Hypothyroidism, also called underactive thyroid or low thyroid, is a disorder of the endocrine system in which the thyroid gland does not produce enough thyroid hormone.', 'Psoriasis': "Psoriasis is a common skin disorder that forms thick, red, bumpy patches covered with silvery scales. They can pop up anywhere, but most appear on the scalp, elbows, knees, and lower back. Psoriasis can't be passed from person to person. It does sometimes happen in members of the same family.", 'GERD': 'Gastroesophageal reflux disease, or GERD, is a digestive disorder that affects the lower esophageal sphincter (LES), the ring of muscle between the esophagus and stomach. Many people, including pregnant women, suffer from heartburn or acid indigestion caused by GERD.', 'Chronic cholestasis': 'Chronic cholestatic diseases, whether occurring in infancy, childhood or adulthood, are characterized by defective bile acid transport from the liver to the intestine, which is caused by primary damage to the biliary epithelium in most cases', 'hepatitis A': "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. The virus is one of several types of hepatitis viruses that cause inflammation and affect your liver's ability to function.", 'Osteoarthristis': 'Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of your bones wears down over time.', '(vertigo) Paroymsal  Positional Vertigo': "Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — the sudden sensation that you're spinning or that the inside of your head is spinning. Benign paroxysmal positional vertigo causes brief episodes of mild to intense dizziness.", 'Hypoglycemia': " Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. Glucose is your body's main energy source. Hypoglycemia is often related to diabetes treatment. But other drugs and a variety of conditions — many rare — can cause low blood sugar in people who don't have diabetes.", 'Acne': 'Acne vulgaris is the formation of comedones, papules, pustules, nodules, and/or cysts as a result of obstruction and inflammation of pilosebaceous units (hair follicles and their accompanying sebaceous gland). Acne develops on the face and upper trunk. It most often affects adolescents.', 'Diabetes': 'Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.', 'Impetigo': "Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects infants and children. Impetigo usually appears as red sores on the face, especially around a child's nose and mouth, and on hands and feet. The sores burst and develop honey-colored crusts.", 'Hypertension': 'Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms.', 'Peptic ulcer diseae': 'Peptic ulcer disease (PUD) is a break in the inner lining of the stomach, the first part of the small intestine, or sometimes the lower esophagus. An ulcer in the stomach is called a gastric ulcer, while one in the first part of the intestines is a duodenal ulcer.', 'Dimorphic hemmorhoids(piles)': 'Hemorrhoids, also spelled haemorrhoids, are vascular structures in the anal canal. In their ... Other names, Haemorrhoids, piles, hemorrhoidal disease .', 'Common Cold': "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold.", 'Chicken pox': 'Chickenpox is a highly contagious disease caused by the varicella-zoster virus (VZV). It can cause an itchy, blister-like rash. The rash first appears on the chest, back, and face, and then spreads over the entire body, causing between 250 and 500 itchy blisters.', 'Cervical spondylosis': 'Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs).', 'Hyperthyroidism': "Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the hormone thyroxine. Hyperthyroidism can accelerate your body's metabolism, causing unintentional weight loss and a rapid or irregular heartbeat.", 'Urinary tract infection':
        'Urinary tract infection: An infection of the kidney, ureter, bladder, or urethra. Abbreviated UTI. Not everyone with a UTI has symptoms, but common symptoms include a frequent urge to urinate and pain or burning when urinating.', 'Varicose veins': 'A vein that has enlarged and twisted, often appearing as a bulging, blue blood vessel that is clearly visible through the skin. Varicose veins are most common in older adults, particularly women, and occur especially on the legs.', 'AIDS': "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). By damaging your immune system, HIV interferes with your body's ability to fight infection and disease.", 'Paralysis (brain hemorrhage)': 'Intracerebral hemorrhage (ICH) is when blood suddenly bursts into brain tissue, causing damage to your brain. Symptoms usually appear suddenly during ICH. They include headache, weakness, confusion, and paralysis, particularly on one side of your body.', 'Typhoid': 'An acute illness characterized by fever caused by infection with the bacterium Salmonella typhi. Typhoid fever has an insidious onset, with fever, headache, constipation, malaise, chills, and muscle pain. Diarrhea is uncommon, and vomiting is not usually severe.', 'Hepatitis B': "Hepatitis B is an infection of your liver. It can cause scarring of the organ, liver failure, and cancer. It can be fatal if it isn't treated. It's spread when people come in contact with the blood, open sores, or body fluids of someone who has the hepatitis B virus.", 'Fungal infection': 'In humans, fungal infections occur when an invading fungus takes over an area of the body and is too much for the immune system to handle. Fungi can live in the air, soil, water, and plants. There are also some fungi that live naturally in the human body. Like many microbes, there are helpful fungi and harmful fungi.', 'Hepatitis C': 'Inflammation of the liver due to the hepatitis C virus (HCV), which is usually spread via blood transfusion (rare), hemodialysis, and needle sticks. The damage hepatitis C does to the liver can lead to cirrhosis and its complications as well as cancer.', 'Migraine': "A migraine can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities.", 'Bronchial Asthma': 'Bronchial asthma is a medical condition which causes the airway path of the lungs to swell and narrow. Due to this swelling, the air path produces excess mucus making it hard to breathe, which results in coughing, short breath, and wheezing. The disease is chronic and interferes with daily working.', 'Alcoholic hepatitis': "Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over an extended period of time. It's also aggravated by binge drinking and ongoing alcohol use. If you develop this condition, you must stop drinking alcohol", 'Jaundice': 'Yellow staining of the skin and sclerae (the whites of the eyes) by abnormally high blood levels of the bile pigment bilirubin. The yellowing extends to other tissues and body fluids. Jaundice was once called the "morbus regius" (the regal disease) in the belief that only the touch of a king could cure it', 'Hepatitis E': 'A rare form of liver inflammation caused by infection with the hepatitis E virus (HEV). It is transmitted via food or drink handled by an infected person or through infected water supplies in areas where fecal matter may get into the water. Hepatitis E does not cause chronic liver disease.', 'Dengue': 'an acute infectious disease caused by a flavivirus (species Dengue virus of the genus Flavivirus), transmitted by aedes mosquitoes, and characterized by headache, severe joint pain, and a rash. — called also breakbone fever, dengue fever.', 'Hepatitis D': 'Hepatitis D, also known as the hepatitis delta virus, is an infection that causes the liver to become inflamed. This swelling can impair liver function and cause long-term liver problems, including liver scarring and cancer. The condition is caused by the hepatitis D virus (HDV).', 'Heart attack': 'The death of heart muscle due to the loss of blood supply. The loss of blood supply is usually caused by a complete blockage of a coronary artery, one of the arteries that supplies blood to the heart muscle.', 'Pneumonia': 'Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe.', 'Arthritis': 'Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis.', 'Gastroenteritis': 'Gastroenteritis is an inflammation of the digestive tract, particularly the stomach, and large and small intestines. Viral and bacterial gastroenteritis are intestinal infections associated with symptoms of diarrhea , abdominal cramps, nausea , and vomiting .', 'Tuberculosis': 'Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis (MTB) bacteria. Tuberculosis generally affects the lungs, but can also affect other parts of the body. Most infections show no symptoms, in which case it is known as latent tuberculosis.'}
pre1 = {'Drug Reaction': 'Stop irritation', 'Malaria': 'Consult nearest hospital', 'Allergy': 'Apply calamine', 'Hypothyroidism': 'Reduce stress', 'Psoriasis': 'Wash hands with warm soapy water', 'GERD': 'Avoid fatty spicy food', 'Chronic cholestasis': 'Cold baths', 'hepatitis A': 'Consult nearest hospital', 'Osteoarthristis': 'Acetaminophen', '(vertigo) Paroymsal  Positional Vertigo': 'Lie down', 'Hypoglycemia': 'Lie down on side', 'Acne': 'Bath twice', 'Diabetes': 'Have balanced diet', 'Impetigo': 'Soak affected area in warm water', 'Hypertension': 'Meditation', 'Peptic ulcer diseae': 'Avoid fatty spicy food', 'Dimorphic hemmorhoids(piles)': 'Avoid fatty spicy food', 'Common Cold': 'Drink vitamin c rich drinks', 'Chicken pox': 'Use neem in bathing ', 'Cervical spondylosis': 'Use heating pad or cold pack',
        'Hyperthyroidism': 'Eat healthy', 'Urinary tract infection': 'Drink plenty of water', 'Varicose veins': 'Lie down flat and raise the leg high', 'AIDS': 'Avoid open cuts', 'Paralysis (brain hemorrhage)': 'Massage', 'Typhoid': 'Eat high calorie vegitables', 'Hepatitis B': 'Consult nearest hospital', 'Fungal infection': 'Bath twice', 'Hepatitis C': 'Consult nearest hospital', 'Migraine': 'Meditation', 'Bronchial Asthma': 'Switch to loose cloothing', 'Alcoholic hepatitis': 'Stop alcohol consumption', 'Jaundice': 'Drink plenty of water', 'Hepatitis E': 'Stop alcohol consumption', 'Dengue': 'Drink papaya leaf juice', 'Hepatitis D': 'Consult doctor', 'Heart attack': 'Manage Stress', 'Pneumonia': 'Consult doctor', 'Arthritis': 'Exercise', 'Gastroenteritis': 'Stop eating solid food for while', 'Tuberculosis': 'Cover mouth'}
pre2 = {'Drug Reaction': 'Consult nearest hospital', 'Malaria': 'Avoid oily food', 'Allergy': 'Cover area with bandage', 'Hypothyroidism': 'Exercise', 'Psoriasis': 'Stop bleeding using pressure', 'GERD': 'Avoid lying down after eating', 'Chronic cholestasis': 'Anti itch medicine', 'hepatitis A': 'Wash hands through', 'Osteoarthristis': 'Consult nearest hospital', '(vertigo) Paroymsal  Positional Vertigo': 'Avoid sudden change in body', 'Hypoglycemia': 'Check in pulse', 'Acne': 'Avoid fatty spicy food', 'Diabetes': 'Exercise', 'Impetigo': 'Use antibiotics', 'Hypertension': 'Salt baths', 'Peptic ulcer diseae': 'Consume probiotic food', 'Dimorphic hemmorhoids(piles)': 'Consume witch hazel', 'Common Cold': 'Take vapour', 'Chicken pox': 'Consume neem leaves', 'Cervical spondylosis': 'Exercise',
        'Hyperthyroidism': 'Massage', 'Urinary tract infection': 'Increase vitamin c intake', 'Varicose veins': 'Use oinments', 'AIDS': 'Wear ppe if possible', 'Paralysis (brain hemorrhage)': 'Eat healthy', 'Typhoid': 'Antiboitic therapy', 'Hepatitis B': 'Vaccination', 'Fungal infection': 'Use detol or neem in bathing water', 'Hepatitis C': 'Vaccination', 'Migraine': 'Reduce stress', 'Bronchial Asthma': 'Take deep breaths', 'Alcoholic hepatitis': 'Consult doctor', 'Jaundice': 'Consume milk thistle', 'Hepatitis E': 'Rest', 'Dengue': 'Avoid fatty spicy food', 'Hepatitis D': 'Medication', 'Heart attack': 'Get proper sleep', 'Pneumonia': 'Medication', 'Arthritis': 'Use hot and cold therapy', 'Gastroenteritis': 'Try taking small sips of water', 'Tuberculosis': 'Consult doctor'}
pre3 = {'Drug Reaction': 'Stop taking drug', 'Malaria': 'Avoid non veg food', 'Allergy': 'Avoid your allergens', 'Hypothyroidism': 'Eat healthy', 'Psoriasis': 'Consult doctor', 'GERD': 'Maintain healthy weight', 'Chronic cholestasis': 'Consult doctor', 'hepatitis A': 'Avoid fatty spicy food', 'Osteoarthristis': 'Follow up', '(vertigo) Paroymsal  Positional Vertigo': 'Avoid abrupt head movment', 'Hypoglycemia': 'Drink sugary drinks', 'Acne': 'Drink plenty of water', 'Diabetes': 'Consult doctor', 'Impetigo': 'Remove scabs with wet compressed cloth', 'Hypertension': 'Reduce stress', 'Peptic ulcer diseae': 'Eliminate milk', 'Dimorphic hemmorhoids(piles)': 'Warm bath with epsom salt', 'Common Cold': 'Avoid cold food', 'Chicken pox': 'Take vaccine', 'Cervical spondylosis': 'Take otc pain reliver',
        'Hyperthyroidism': 'Use lemon balm', 'Urinary tract infection': 'Drink cranberry juice', 'Varicose veins': 'Use vein compression', 'AIDS': 'Consult doctor', 'Paralysis (brain hemorrhage)': 'Exercise', 'Typhoid': 'Consult doctor', 'Hepatitis B': 'Eat healthy', 'Fungal infection': 'Keep infected area dry', 'Hepatitis C': 'Eat healthy', 'Migraine': 'Use poloroid glasses in sun', 'Bronchial Asthma': 'Get away from trigger', 'Alcoholic hepatitis': 'Medication', 'Jaundice': 'Eat fruits and high fiberous food', 'Hepatitis E': 'Consult doctor', 'Dengue': 'Keep mosquitos away', 'Hepatitis D': 'Eat healthy', 'Heart attack': 'Keep check and control the cholesterol levels', 'Pneumonia': 'Rest', 'Arthritis': 'Try acupuncture', 'Gastroenteritis': 'Rest', 'Tuberculosis': 'Medication'}
pre4 = {'Drug Reaction': 'Follow up', 'Malaria': 'Keep mosquitos out', 'Allergy': 'Use ice to compress itching', 'Hypothyroidism': 'Get proper sleep', 'Psoriasis': 'Salt baths', 'GERD': 'Exercise', 'Chronic cholestasis': 'Eat healthy', 'hepatitis A': 'Medication', 'Osteoarthristis': 'Salt baths', '(vertigo) Paroymsal  Positional Vertigo': 'Relax', 'Hypoglycemia': 'Consult doctor', 'Acne': 'Avoid too many products', 'Diabetes': 'Follow up', 'Impetigo': 'Consult doctor', 'Hypertension': 'Get proper sleep', 'Peptic ulcer diseae': 'Limit alcohol', 'Dimorphic hemmorhoids(piles)': 'Consume alovera juice', 'Common Cold': 'Keep fever in check', 'Chicken pox': 'Avoid public places', 'Cervical spondylosis': 'Consult doctor',
        'Hyperthyroidism': 'Take radioactive iodine treatment', 'Urinary tract infection': 'Take probiotics', 'Varicose veins': 'Dont stand still for long', 'AIDS': 'Follow up', 'Paralysis (brain hemorrhage)': 'Consult doctor', 'Typhoid': 'Medication', 'Hepatitis B': 'Medication', 'Fungal infection': 'Use clean cloths', 'Hepatitis C': 'Medication', 'Migraine': 'Consult doctor', 'Bronchial Asthma': 'Seek help', 'Alcoholic hepatitis': 'Follow up', 'Jaundice': 'Medication', 'Hepatitis E': 'Medication', 'Dengue': 'Keep hydrated', 'Hepatitis D': 'Follow up', 'Heart attack': 'Exercise and be physically fit', 'Pneumonia': 'Follow up', 'Arthritis': 'Massage', 'Gastroenteritis': 'Ease back into eating', 'Tuberculosis': 'Rest'}


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def marks():
    print("ob")
    if request.method == 'POST':
        f = request.form.getlist('symptom')
        headache = overweight = alcohol = fever = runny = ""
        headache = request.form.getlist('headache')
        if len(headache):
            headache = headache[0]
        overweight = request.form.getlist('overweight')
        if len(overweight):
            overweight = overweight[0]
        alcohol = request.form.getlist('alcohol')
        if len(alcohol):
            alcohol = alcohol[0]
        fever = request.form.getlist('fever')
        if len(fever):
            fever = fever[0]
        runny = request.form.getlist('runny nose')
        if len(runny):
            runny = runny[0]

        cols = [0]*131
        if(headache == "Yes"):
            idx = unique.index("headache")
            cols[idx] = 1

        if(overweight == "Yes"):
            idx = unique.index("obesity")
            cols[idx] = 1

        if(fever == "Yes"):
            idx = unique.index("high fever")
            cols[idx] = 1

        if(runny == "Yes"):
            idx = unique.index("runny nose")
            cols[idx] = 1

        if(alcohol == "Yes"):
            idx = unique.index("history of alcohol consumption")
            cols[idx] = 1

        for i in range(0, len(unique)):
            if unique[i] in f:
                cols[i] = 1

        pred = rfc.predict([cols])
        pred = pred[0]
        pred = pred.strip()
        print(pred)
        print(desc['Hypertension'])
        des = desc[pred]
        p1 = pre1[pred]
        p2 = pre2[pred]
        p3 = pre3[pred]
        p4 = pre4[pred]
        t = test1[pred]

        cnt = 0
        for i in range(0, len(cols)):
            if cols[i] == 1:
                cnt += 1

        if cnt < 5:
            pred = ""

    return render_template("ind.html", pred=pred, des=des, pre1=p1, pre2=p2, pre3=p3, pre4=p4, test=t)


if __name__ == '__main__':
    app.run(debug=True)


# disease=pred, des=des, pre1=p1, pre2=p2, pre3=p3, pre4=p4
