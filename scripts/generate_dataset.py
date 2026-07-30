"""
CareBot AI - Dataset Generation Script
Builds a symptom-disease training dataset plus disease reference info
(descriptions, precautions, severity) used by the ML model and the
recommendation engine.

The symptom-disease associations below reflect commonly known, general
medical knowledge (the same structure used by widely published open
symptom-checker datasets) and are for educational/demo purposes only.
CareBot is NOT a diagnostic tool and does not replace professional care.
"""
import csv
import json
import random
import os

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Disease -> core symptom sets
# ---------------------------------------------------------------------------
DISEASE_SYMPTOMS = {
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "mild_cough", "congestion", "mild_fever"],
    "Influenza": ["high_fever", "body_ache", "chills", "fatigue", "headache", "dry_cough", "sore_throat"],
    "COVID-19": ["fever", "dry_cough", "fatigue", "loss_of_taste", "loss_of_smell", "shortness_of_breath", "body_ache"],
    "Migraine": ["severe_headache", "nausea", "sensitivity_to_light", "sensitivity_to_sound", "visual_aura", "vomiting"],
    "Tension Headache": ["mild_headache", "neck_stiffness", "scalp_tenderness", "fatigue", "irritability"],
    "Gastroenteritis": ["diarrhea", "vomiting", "abdominal_cramps", "nausea", "mild_fever", "dehydration"],
    "Food Poisoning": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "weakness"],
    "Acid Reflux (GERD)": ["heartburn", "regurgitation", "chest_burning", "bloating", "sour_taste", "difficulty_swallowing"],
    "Peptic Ulcer": ["abdominal_pain", "bloating", "nausea", "burning_stomach_pain", "weight_loss", "indigestion"],
    "Irritable Bowel Syndrome": ["abdominal_cramps", "bloating", "diarrhea", "constipation", "gas", "mucus_in_stool"],
    "Urinary Tract Infection": ["burning_urination", "frequent_urination", "cloudy_urine", "pelvic_pain", "mild_fever"],
    "Kidney Stones": ["severe_flank_pain", "blood_in_urine", "nausea", "vomiting", "painful_urination", "groin_pain"],
    "Bronchitis": ["persistent_cough", "mucus_production", "chest_discomfort", "mild_fever", "fatigue", "wheezing"],
    "Pneumonia": ["high_fever", "productive_cough", "chest_pain", "shortness_of_breath", "chills", "fatigue"],
    "Asthma": ["wheezing", "shortness_of_breath", "chest_tightness", "coughing", "difficulty_breathing"],
    "Allergic Rhinitis": ["sneezing", "itchy_eyes", "runny_nose", "nasal_congestion", "watery_eyes"],
    "Sinusitis": ["facial_pain", "nasal_congestion", "headache", "thick_nasal_discharge", "reduced_smell"],
    "Conjunctivitis": ["red_eyes", "itchy_eyes", "watery_eyes", "eye_discharge", "gritty_feeling"],
    "Dengue Fever": ["high_fever", "severe_headache", "joint_pain", "muscle_pain", "rash", "nausea", "pain_behind_eyes"],
    "Malaria": ["high_fever", "chills", "sweating", "headache", "nausea", "muscle_pain", "fatigue"],
    "Typhoid Fever": ["prolonged_fever", "weakness", "abdominal_pain", "headache", "loss_of_appetite", "constipation"],
    "Chickenpox": ["itchy_rash", "fluid_filled_blisters", "mild_fever", "fatigue", "loss_of_appetite"],
    "Measles": ["high_fever", "cough", "runny_nose", "red_eyes", "rash", "koplik_spots"],
    "Hypertension": ["headache", "dizziness", "blurred_vision", "chest_pain", "shortness_of_breath", "nosebleed"],
    "Type 2 Diabetes": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "slow_healing_sores", "weight_loss"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_intolerance", "dry_skin", "constipation", "hair_loss"],
    "Hyperthyroidism": ["weight_loss", "rapid_heartbeat", "sweating", "nervousness", "tremor", "heat_intolerance"],
    "Anemia": ["fatigue", "pale_skin", "shortness_of_breath", "dizziness", "cold_hands", "weakness"],
    "Anxiety Disorder": ["excessive_worry", "restlessness", "rapid_heartbeat", "sweating", "difficulty_concentrating", "irritability"],
    "Depression": ["persistent_sadness", "loss_of_interest", "fatigue", "sleep_disturbance", "appetite_change", "difficulty_concentrating"],
    "Insomnia": ["difficulty_falling_asleep", "waking_at_night", "daytime_fatigue", "irritability", "difficulty_concentrating"],
    "Eczema": ["itchy_skin", "dry_skin", "red_rash", "skin_cracking", "inflamed_skin"],
    "Psoriasis": ["red_patches", "silvery_scales", "itchy_skin", "dry_cracked_skin", "joint_pain"],
    "Acne": ["pimples", "blackheads", "oily_skin", "skin_inflammation", "scarring"],
    "Fungal Skin Infection": ["itchy_skin", "red_rash", "scaly_skin", "skin_peeling", "burning_sensation"],
    "Rheumatoid Arthritis": ["joint_pain", "joint_swelling", "morning_stiffness", "fatigue", "warm_joints"],
    "Osteoarthritis": ["joint_pain", "joint_stiffness", "reduced_flexibility", "swelling", "grating_sensation"],
    "Gout": ["sudden_joint_pain", "joint_swelling", "redness", "warmth_in_joint", "limited_range_of_motion"],
    "Chronic Fatigue Syndrome": ["persistent_fatigue", "muscle_pain", "headache", "poor_sleep", "difficulty_concentrating"],
    "Appendicitis": ["abdominal_pain", "loss_of_appetite", "nausea", "vomiting", "fever", "pain_moving_to_right_side"],
}

SEVERITY = {
    "Common Cold": "Low", "Allergic Rhinitis": "Low", "Tension Headache": "Low",
    "Conjunctivitis": "Low", "Acne": "Low", "Eczema": "Low", "Insomnia": "Low",
    "Fungal Skin Infection": "Low", "Sinusitis": "Low",
    "Influenza": "Medium", "Migraine": "Medium", "Gastroenteritis": "Medium",
    "Food Poisoning": "Medium", "Acid Reflux (GERD)": "Medium", "Irritable Bowel Syndrome": "Medium",
    "Urinary Tract Infection": "Medium", "Bronchitis": "Medium", "Asthma": "Medium",
    "Chickenpox": "Medium", "Hypothyroidism": "Medium", "Hyperthyroidism": "Medium",
    "Anemia": "Medium", "Anxiety Disorder": "Medium", "Depression": "Medium",
    "Psoriasis": "Medium", "Rheumatoid Arthritis": "Medium", "Osteoarthritis": "Medium",
    "Gout": "Medium", "Chronic Fatigue Syndrome": "Medium", "Peptic Ulcer": "Medium",
    "Hypertension": "Medium",
    "COVID-19": "High", "Pneumonia": "High", "Dengue Fever": "High", "Malaria": "High",
    "Typhoid Fever": "High", "Measles": "High", "Kidney Stones": "High",
    "Type 2 Diabetes": "High", "Appendicitis": "High",
}

DESCRIPTIONS = {
    "Common Cold": "A mild viral infection of the nose and throat, usually resolving on its own within 7-10 days.",
    "Influenza": "A contagious respiratory illness caused by influenza viruses, more severe than the common cold.",
    "COVID-19": "An infectious disease caused by the SARS-CoV-2 virus, affecting the respiratory system with a wide range of severity.",
    "Migraine": "A neurological condition causing intense, throbbing headaches often with nausea and light sensitivity.",
    "Tension Headache": "The most common type of headache, causing mild to moderate dull, aching pain.",
    "Gastroenteritis": "Inflammation of the stomach and intestines, typically due to viral or bacterial infection.",
    "Food Poisoning": "Illness caused by consuming contaminated food, leading to gastrointestinal distress.",
    "Acid Reflux (GERD)": "A digestive disorder where stomach acid frequently flows back into the esophagus.",
    "Peptic Ulcer": "Open sores that develop on the inner lining of the stomach or upper small intestine.",
    "Irritable Bowel Syndrome": "A common disorder affecting the large intestine, causing cramping and altered bowel habits.",
    "Urinary Tract Infection": "An infection in any part of the urinary system, most commonly the bladder and urethra.",
    "Kidney Stones": "Hard deposits of minerals and salts that form inside the kidneys, causing significant pain when passing.",
    "Bronchitis": "Inflammation of the lining of the bronchial tubes, which carry air to and from the lungs.",
    "Pneumonia": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid.",
    "Asthma": "A chronic condition in which airways narrow, swell, and produce extra mucus, making breathing difficult.",
    "Allergic Rhinitis": "An allergic response causing cold-like symptoms, such as sneezing and a runny nose.",
    "Sinusitis": "Inflammation or swelling of the tissue lining the sinuses.",
    "Conjunctivitis": "Inflammation of the membrane covering the eye, commonly known as pink eye.",
    "Dengue Fever": "A mosquito-borne viral infection causing severe flu-like illness.",
    "Malaria": "A mosquito-borne disease caused by a parasite, producing recurring fever and chills.",
    "Typhoid Fever": "A bacterial infection spread through contaminated food and water, causing prolonged fever.",
    "Chickenpox": "A highly contagious viral infection causing an itchy, blister-like rash.",
    "Measles": "A highly contagious viral infection of the respiratory system, causing a distinctive rash.",
    "Hypertension": "Persistently elevated blood pressure in the arteries, a major cardiovascular risk factor.",
    "Type 2 Diabetes": "A chronic condition affecting how the body processes blood sugar (glucose).",
    "Hypothyroidism": "A condition where the thyroid gland doesn't produce enough thyroid hormone.",
    "Hyperthyroidism": "A condition where the thyroid gland produces too much thyroid hormone.",
    "Anemia": "A condition marked by a deficiency of red blood cells or hemoglobin, reducing oxygen delivery.",
    "Anxiety Disorder": "A mental health condition characterized by persistent, excessive worry.",
    "Depression": "A mood disorder causing persistent feelings of sadness and loss of interest.",
    "Insomnia": "A sleep disorder characterized by difficulty falling or staying asleep.",
    "Eczema": "A condition that makes skin red, inflamed, and itchy.",
    "Psoriasis": "A skin disease causing red, itchy scaly patches, most commonly on the knees, elbows, and scalp.",
    "Acne": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
    "Fungal Skin Infection": "A skin infection caused by fungi, thriving in warm, moist areas of the body.",
    "Rheumatoid Arthritis": "An autoimmune disorder causing joint inflammation, pain, and swelling.",
    "Osteoarthritis": "A degenerative joint disease caused by the breakdown of joint cartilage.",
    "Gout": "A form of inflammatory arthritis caused by excess uric acid, causing sudden, severe joint pain.",
    "Chronic Fatigue Syndrome": "A complex disorder characterized by extreme fatigue not explained by an underlying condition.",
    "Appendicitis": "Inflammation of the appendix, a medical emergency usually requiring surgery.",
}

PRECAUTIONS = {
    "Common Cold": ["Rest and stay hydrated", "Use a humidifier", "Gargle warm salt water", "Avoid close contact with others"],
    "Influenza": ["Rest and stay hydrated", "Take fever reducers as directed", "Avoid contact with others", "See a doctor if symptoms worsen"],
    "COVID-19": ["Isolate from others", "Monitor oxygen levels", "Stay hydrated and rest", "Seek emergency care for breathing difficulty"],
    "Migraine": ["Rest in a dark, quiet room", "Apply a cold compress", "Stay hydrated", "Avoid known triggers"],
    "Tension Headache": ["Rest and reduce stress", "Apply a warm or cold compress", "Stay hydrated", "Practice relaxation techniques"],
    "Gastroenteritis": ["Stay hydrated with oral rehydration solutions", "Eat bland foods (BRAT diet)", "Rest", "Avoid dairy and fatty foods"],
    "Food Poisoning": ["Stay hydrated", "Rest the stomach with bland foods", "Avoid solid food until vomiting stops", "Seek care if symptoms are severe"],
    "Acid Reflux (GERD)": ["Avoid trigger foods", "Eat smaller meals", "Avoid lying down after eating", "Elevate the head while sleeping"],
    "Peptic Ulcer": ["Avoid NSAIDs and alcohol", "Eat smaller, frequent meals", "Avoid spicy foods", "Consult a doctor for evaluation"],
    "Irritable Bowel Syndrome": ["Identify and avoid trigger foods", "Manage stress", "Increase fiber gradually", "Stay hydrated"],
    "Urinary Tract Infection": ["Drink plenty of water", "Urinate frequently, don't hold it in", "Avoid irritants like caffeine", "See a doctor for antibiotics"],
    "Kidney Stones": ["Drink plenty of water", "Reduce salt intake", "Take pain relief as directed", "Seek medical care for severe pain"],
    "Bronchitis": ["Rest and stay hydrated", "Use a humidifier", "Avoid smoke and irritants", "See a doctor if cough persists over 3 weeks"],
    "Pneumonia": ["Seek medical attention promptly", "Rest and stay hydrated", "Take prescribed antibiotics fully", "Monitor breathing closely"],
    "Asthma": ["Avoid known triggers", "Keep rescue inhaler accessible", "Follow an asthma action plan", "Seek emergency care for severe attacks"],
    "Allergic Rhinitis": ["Avoid known allergens", "Use air purifiers", "Consider antihistamines", "Keep windows closed during high pollen"],
    "Sinusitis": ["Use saline nasal rinses", "Stay hydrated", "Apply warm compresses to the face", "See a doctor if symptoms persist beyond 10 days"],
    "Conjunctivitis": ["Avoid touching or rubbing eyes", "Wash hands frequently", "Use clean, separate towels", "See a doctor if vision is affected"],
    "Dengue Fever": ["Rest and stay hydrated", "Avoid aspirin/ibuprofen (use acetaminophen)", "Monitor for warning signs", "Seek immediate care for bleeding or severe pain"],
    "Malaria": ["Seek medical care immediately for antimalarial treatment", "Stay hydrated", "Rest", "Use mosquito protection to prevent spread"],
    "Typhoid Fever": ["Seek medical care for antibiotics", "Stay hydrated", "Eat soft, easily digestible foods", "Practice strict hand hygiene"],
    "Chickenpox": ["Avoid scratching the rash", "Keep skin clean and cool", "Isolate to prevent spread", "Trim nails to reduce skin damage"],
    "Measles": ["Isolate to prevent spread", "Rest and stay hydrated", "Manage fever", "Seek care for breathing difficulty"],
    "Hypertension": ["Reduce salt intake", "Exercise regularly", "Manage stress", "Monitor blood pressure and take medication as prescribed"],
    "Type 2 Diabetes": ["Monitor blood sugar regularly", "Follow a balanced diet", "Exercise regularly", "Take medication as prescribed"],
    "Hypothyroidism": ["Take thyroid medication consistently", "Get regular thyroid level checks", "Maintain a balanced diet", "Report new symptoms to your doctor"],
    "Hyperthyroidism": ["Follow prescribed treatment plan", "Avoid excess iodine", "Manage stress", "Get regular thyroid monitoring"],
    "Anemia": ["Eat iron-rich foods", "Take supplements as prescribed", "Get evaluated for underlying cause", "Avoid excessive fatigue"],
    "Anxiety Disorder": ["Practice relaxation and breathing exercises", "Limit caffeine", "Maintain a regular sleep schedule", "Consider speaking with a mental health professional"],
    "Depression": ["Reach out to a mental health professional", "Maintain social connections", "Keep a regular routine", "Engage in physical activity"],
    "Insomnia": ["Maintain a consistent sleep schedule", "Limit screen time before bed", "Avoid caffeine late in the day", "Create a relaxing bedtime routine"],
    "Eczema": ["Moisturize regularly", "Avoid known irritants", "Use mild, fragrance-free soap", "Avoid scratching"],
    "Psoriasis": ["Moisturize skin daily", "Avoid triggers like stress and injury", "Get moderate sun exposure", "Follow prescribed topical treatment"],
    "Acne": ["Wash face gently twice daily", "Avoid picking at blemishes", "Use non-comedogenic products", "See a dermatologist for persistent acne"],
    "Fungal Skin Infection": ["Keep skin clean and dry", "Avoid sharing personal items", "Use antifungal treatment as directed", "Wear breathable clothing"],
    "Rheumatoid Arthritis": ["Stay active with low-impact exercise", "Manage stress", "Follow prescribed treatment plan", "Get regular rest"],
    "Osteoarthritis": ["Maintain a healthy weight", "Engage in low-impact exercise", "Use joint support aids if needed", "Apply heat/cold therapy"],
    "Gout": ["Limit purine-rich foods and alcohol", "Stay hydrated", "Take prescribed medication during flares", "Rest the affected joint"],
    "Chronic Fatigue Syndrome": ["Pace activities to avoid overexertion", "Prioritize sleep hygiene", "Manage stress", "Work with a doctor on a management plan"],
    "Appendicitis": ["Seek emergency medical care immediately", "Avoid eating or drinking until evaluated", "Do not take laxatives or pain medication that could mask symptoms"],
}

# Supportive, commonly-known home comfort measures shown alongside the
# prediction. For serious/emergency conditions these emphasise that home
# care is only supportive and professional care is needed.
HOME_REMEDIES = {
    "Common Cold": ["Sip warm honey and lemon tea to soothe the throat", "Inhale steam from a bowl of hot water", "Gargle with warm salt water", "Rest and drink plenty of warm fluids"],
    "Influenza": ["Drink warm fluids like ginger or herbal tea", "Take honey to ease cough and sore throat", "Use a warm compress for body aches", "Rest fully and keep hydrated"],
    "COVID-19": ["Stay well hydrated with warm fluids", "Use steam inhalation to ease congestion", "Gargle warm salt water for a sore throat", "Rest and monitor your symptoms closely"],
    "Migraine": ["Rest in a dark, quiet room", "Apply a cold compress to the forehead", "Sip water to stay hydrated", "Gently massage the temples and neck"],
    "Tension Headache": ["Apply a warm compress to the neck and shoulders", "Practice deep breathing or meditation", "Stay hydrated", "Take short breaks away from screens"],
    "Gastroenteritis": ["Sip oral rehydration solution or coconut water", "Eat bland foods like bananas, rice, and toast", "Drink ginger tea to calm nausea", "Rest and avoid dairy"],
    "Food Poisoning": ["Sip small amounts of water or ORS frequently", "Try ginger or peppermint tea for nausea", "Eat plain foods once vomiting stops", "Rest and avoid solid food early on"],
    "Acid Reflux (GERD)": ["Sip cold milk or chamomile tea", "Chew a small piece of fresh ginger", "Avoid lying down for 2-3 hours after meals", "Eat smaller, slower meals"],
    "Peptic Ulcer": ["Drink cabbage juice or coconut water", "Take a little honey on an empty stomach", "Avoid spicy and acidic foods", "Include probiotic foods like yogurt"],
    "Irritable Bowel Syndrome": ["Sip peppermint tea to ease cramps", "Apply a warm compress to the abdomen", "Add fiber-rich foods gradually", "Practice stress-reducing relaxation"],
    "Urinary Tract Infection": ["Drink plenty of water to flush the system", "Try unsweetened cranberry juice", "Apply a warm heating pad to the lower abdomen", "Urinate frequently and don't hold it in"],
    "Kidney Stones": ["Drink lots of water throughout the day", "Try lemon water to help break down stones", "Apply a warm compress for pain relief", "Reduce salt in your diet"],
    "Bronchitis": ["Inhale steam to loosen mucus", "Drink warm honey and lemon", "Use a humidifier at night", "Stay hydrated and rest your voice"],
    "Pneumonia": ["Rest completely and drink warm fluids", "Use steam inhalation to ease breathing", "Take warm honey for cough comfort", "Seek medical care — home care is only supportive"],
    "Asthma": ["Sit upright and practice slow breathing", "Sip warm water or ginger tea", "Avoid triggers like dust and smoke", "Keep your rescue inhaler nearby — seek care for severe attacks"],
    "Allergic Rhinitis": ["Rinse nasal passages with saline", "Inhale steam to clear congestion", "Try a spoon of local honey", "Keep windows closed during high pollen"],
    "Sinusitis": ["Use steam inhalation with a few drops of eucalyptus", "Rinse sinuses with a saline solution", "Apply a warm compress to the face", "Stay well hydrated"],
    "Conjunctivitis": ["Apply a clean warm or cold compress to the eyes", "Rinse eyes gently with cooled boiled water", "Avoid rubbing the eyes", "Use separate clean towels"],
    "Dengue Fever": ["Drink plenty of fluids and ORS", "Try papaya leaf juice to support recovery", "Rest completely", "Use acetaminophen for fever, not aspirin — seek care for warning signs"],
    "Malaria": ["Stay hydrated with fluids and ORS", "Rest fully", "Use a cool compress for fever", "Seek medical care immediately — remedies are only supportive"],
    "Typhoid Fever": ["Drink ORS and plenty of fluids", "Eat soft, easily digestible foods", "Sip ginger or tulsi (basil) tea", "Rest — antibiotics from a doctor are essential"],
    "Chickenpox": ["Take a cool oatmeal or baking soda bath", "Apply calamine lotion to reduce itching", "Keep nails trimmed to avoid scratching", "Wear loose, soft cotton clothing"],
    "Measles": ["Rest in a comfortable, dimly lit room", "Stay hydrated with fluids", "Use a cool compress for fever", "Eat vitamin A-rich foods — seek medical guidance"],
    "Hypertension": ["Reduce salt and eat potassium-rich foods", "Practice deep breathing or meditation", "Take regular brisk walks", "Limit caffeine and alcohol"],
    "Type 2 Diabetes": ["Include fiber-rich, whole foods", "Take a short walk after meals", "Stay hydrated with water", "Add cinnamon or fenugreek to meals (with your doctor's okay)"],
    "Hypothyroidism": ["Eat iodine-rich foods like fish and dairy", "Include selenium sources such as nuts", "Exercise regularly to boost energy", "Take medication consistently as prescribed"],
    "Hyperthyroidism": ["Eat calming, balanced meals", "Limit caffeine and stimulants", "Practice relaxation to manage stress", "Ensure you get adequate rest"],
    "Anemia": ["Eat iron-rich foods like spinach and lentils", "Pair iron foods with vitamin C for absorption", "Include beetroot and dates", "Avoid tea or coffee with meals"],
    "Anxiety Disorder": ["Practice slow, deep breathing exercises", "Try chamomile or green tea", "Exercise or take regular walks", "Limit caffeine and keep a steady sleep routine"],
    "Depression": ["Get regular sunlight and fresh air", "Stay physically active", "Maintain social connection", "Keep a consistent daily routine — seek professional support"],
    "Insomnia": ["Drink warm milk or chamomile tea before bed", "Keep a consistent sleep schedule", "Avoid screens an hour before bed", "Try relaxation or light stretching"],
    "Eczema": ["Moisturize with a fragrance-free cream", "Apply a cool compress to itchy areas", "Take lukewarm oatmeal baths", "Avoid harsh soaps and known irritants"],
    "Psoriasis": ["Moisturize skin generously", "Take lukewarm baths with oatmeal or Epsom salt", "Get brief, moderate sunlight", "Apply aloe vera to affected areas"],
    "Acne": ["Wash the face gently twice daily", "Apply aloe vera or a little honey to spots", "Avoid touching or picking at the skin", "Use a warm compress for painful pimples"],
    "Fungal Skin Infection": ["Keep the affected area clean and dry", "Apply plain yogurt or diluted tea tree oil", "Wear loose, breathable cotton clothing", "Avoid sharing towels and personal items"],
    "Rheumatoid Arthritis": ["Apply warm or cold packs to sore joints", "Do gentle stretching and low-impact exercise", "Add anti-inflammatory foods like turmeric and ginger", "Get adequate rest"],
    "Osteoarthritis": ["Apply heat or cold to painful joints", "Do gentle low-impact exercises", "Maintain a healthy weight", "Try adding turmeric to your diet"],
    "Gout": ["Drink plenty of water to flush uric acid", "Apply a cold compress to the joint", "Eat cherries or drink cherry juice", "Avoid alcohol and purine-rich foods"],
    "Chronic Fatigue Syndrome": ["Pace activities and rest between tasks", "Keep a consistent sleep routine", "Stay hydrated and eat balanced meals", "Try light stretching or gentle yoga"],
    "Appendicitis": ["Do not eat, drink, or take laxatives", "Avoid placing heating pads on the abdomen", "Rest and stay still", "This is an emergency — go to a hospital immediately"],
}

# The medical specialty / department best suited to each condition. Used after
# a prediction to point the user toward the right kind of hospital or clinic.
SPECIALTIES = {
    "Common Cold": "General Medicine",
    "Influenza": "General Medicine",
    "COVID-19": "Pulmonology / Infectious Disease",
    "Migraine": "Neurology",
    "Tension Headache": "Neurology",
    "Gastroenteritis": "Gastroenterology",
    "Food Poisoning": "Gastroenterology",
    "Acid Reflux (GERD)": "Gastroenterology",
    "Peptic Ulcer": "Gastroenterology",
    "Irritable Bowel Syndrome": "Gastroenterology",
    "Urinary Tract Infection": "Urology",
    "Kidney Stones": "Urology",
    "Bronchitis": "Pulmonology",
    "Pneumonia": "Pulmonology",
    "Asthma": "Pulmonology",
    "Allergic Rhinitis": "ENT (Otolaryngology)",
    "Sinusitis": "ENT (Otolaryngology)",
    "Conjunctivitis": "Ophthalmology",
    "Dengue Fever": "Infectious Disease",
    "Malaria": "Infectious Disease",
    "Typhoid Fever": "Infectious Disease",
    "Chickenpox": "Dermatology / Infectious Disease",
    "Measles": "Pediatrics / Infectious Disease",
    "Hypertension": "Cardiology",
    "Type 2 Diabetes": "Endocrinology",
    "Hypothyroidism": "Endocrinology",
    "Hyperthyroidism": "Endocrinology",
    "Anemia": "Hematology",
    "Anxiety Disorder": "Psychiatry / Mental Health",
    "Depression": "Psychiatry / Mental Health",
    "Insomnia": "Sleep Medicine",
    "Eczema": "Dermatology",
    "Psoriasis": "Dermatology",
    "Acne": "Dermatology",
    "Fungal Skin Infection": "Dermatology",
    "Rheumatoid Arthritis": "Rheumatology",
    "Osteoarthritis": "Orthopedics",
    "Gout": "Rheumatology",
    "Chronic Fatigue Syndrome": "General Medicine",
    "Appendicitis": "Emergency & General Surgery",
}

# ---------------------------------------------------------------------------
# 2. Build the master symptom list
# ---------------------------------------------------------------------------
all_symptoms = sorted({s for symptoms in DISEASE_SYMPTOMS.values() for s in symptoms})
print(f"Diseases: {len(DISEASE_SYMPTOMS)}  |  Unique symptoms: {len(all_symptoms)}")

# ---------------------------------------------------------------------------
# 3. Generate synthetic training rows
#    Each row = a symptom presentation for a disease (a subset of its core
#    symptoms, +/- a little noise) so the classifier has to generalize
#    instead of doing exact lookup.
# ---------------------------------------------------------------------------
ROWS_PER_DISEASE = 150
rows = []
for disease, symptoms in DISEASE_SYMPTOMS.items():
    n = len(symptoms)
    # Real users rarely report every symptom of a condition, so we train on
    # presentations ranging from about half the symptoms up to the full set.
    # This makes the model far more reliable on the sparse inputs it actually
    # sees in production instead of assuming near-complete symptom lists.
    low = max(2, round(n * 0.5))
    for _ in range(ROWS_PER_DISEASE):
        k = random.randint(low, n)
        chosen = set(random.sample(symptoms, k))
        # occasional unrelated "noise" symptom, as happens in real reports
        if random.random() < 0.15:
            noise_pool = [s for s in all_symptoms if s not in symptoms]
            chosen.add(random.choice(noise_pool))
        row = {s: (1 if s in chosen else 0) for s in all_symptoms}
        row["prognosis"] = disease
        rows.append(row)

random.shuffle(rows)

# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------
csv_path = os.path.join(OUT_DIR, "symptom_disease_dataset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_symptoms + ["prognosis"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows -> {csv_path}")

symptoms_path = os.path.join(OUT_DIR, "symptoms_list.json")
with open(symptoms_path, "w") as f:
    json.dump(all_symptoms, f, indent=2)
print(f"Wrote symptom list -> {symptoms_path}")

info_path = os.path.join(OUT_DIR, "disease_info.json")
disease_info = {}
for d in DISEASE_SYMPTOMS:
    disease_info[d] = {
        "description": DESCRIPTIONS.get(d, ""),
        "precautions": PRECAUTIONS.get(d, []),
        "home_remedies": HOME_REMEDIES.get(d, []),
        "specialty": SPECIALTIES.get(d, "General Medicine"),
        "severity": SEVERITY.get(d, "Medium"),
        "core_symptoms": DISEASE_SYMPTOMS[d],
    }
with open(info_path, "w") as f:
    json.dump(disease_info, f, indent=2)
print(f"Wrote disease info -> {info_path}")
