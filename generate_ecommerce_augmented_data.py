import pandas as pd
import random
import os

AUGMENTED_ECOMMERCE_REVIEWS = [
    # Electronics - Mobile
    ("Phone ki battery life bahut mast hai but camera quality thodi bekar hai low light me", 
     ["battery life", "camera quality"], ["positive", "negative"], ["battery_performance", "camera_display"], True),
    ("Display bahut smooth hai 120Hz refresh rate ke sath aur charging speed super fast hai", 
     ["Display", "charging speed"], ["positive", "positive"], ["camera_display", "battery_performance"], True),
    ("Phone jaldi garam ho jata hai gaming ke time par processor performance achhi hai", 
     ["Phone", "processor performance"], ["negative", "positive"], ["product_item", "battery_performance"], True),
    ("Sound quality aur speaker loud hai, build quality bhi solid hai", 
     ["Sound quality", "speaker", "build quality"], ["positive", "positive", "positive"], ["sound_audio", "sound_audio", "quality_build"], True),
    ("Fingerprint sensor bilkul slow hai aur face unlock kaam nahi kar raha", 
     ["Fingerprint sensor", "face unlock"], ["negative", "negative"], ["hardware_sensor", "hardware_sensor"], True),
    ("Price ke hisab se performance lajawab hai, screen colors punchy hain", 
     ["Price", "performance", "screen"], ["positive", "positive", "positive"], ["price_value", "battery_performance", "camera_display"], True),
    ("Camera zoom bekar hai but daylight photography me pictures bahut clear aati hain", 
     ["Camera zoom", "daylight photography"], ["negative", "positive"], ["camera_display", "camera_display"], True),
    ("Battery backup 2 din chalta hai par charger box me nahi mila", 
     ["Battery backup", "charger"], ["positive", "negative"], ["battery_performance", "accessories"], True),
    ("Heavy gaming me frame drop hota hai lekin normal use ke liye phone best hai", 
     ["Heavy gaming", "normal use"], ["negative", "positive"], ["battery_performance", "product_item"], True),
    ("Is price segment me heating issue sabse bada problem hai", 
     ["Price", "heating issue"], ["neutral", "negative"], ["price_value", "battery_performance"], True),
    
    # Electronics - Audio / Earphones
    ("Bass bohot heavy hai aur sound clarity perfect hai lekin ear fit loose hai", 
     ["Bass", "sound clarity", "ear fit"], ["positive", "positive", "negative"], ["sound_audio", "sound_audio", "fit_size"], True),
    ("Mic quality call ke liye bilkul ghatiya hai, samne wale ko voice nahi jati", 
     ["Mic quality", "voice"], ["negative", "negative"], ["sound_audio", "sound_audio"], True),
    ("ANC mode zabardast kaam karta hai travel me aur battery life 30 hours hai", 
     ["ANC mode", "battery life"], ["positive", "positive"], ["sound_audio", "battery_performance"], True),
    ("Bluetooth connectivity baar baar disconnect hoti hai, waste of money", 
     ["Bluetooth connectivity", "money"], ["negative", "negative"], ["connectivity", "price_value"], True),
    ("Earbuds ka design stylish hai aur case lightweight hai", 
     ["design", "case"], ["positive", "positive"], ["design_appearance", "quality_build"], True),
    
    # Laptops & Computing
    ("Laptop ka boot time 10 seconds hai aur keyboard typing experience comfortable hai", 
     ["boot time", "keyboard typing experience"], ["positive", "positive"], ["performance", "hardware"], True),
    ("Trackpad smooth hai lekin display viewing angles acche nahi hain", 
     ["Trackpad", "display viewing angles"], ["positive", "negative"], ["hardware", "camera_display"], True),
    ("Fan noise bahut zyada hai heavy workload me par multitasking speed fast hai", 
     ["Fan noise", "multitasking speed"], ["negative", "positive"], ["hardware", "performance"], True),
    ("RAM aur SSD upgradeable hai jo future-proof banata hai is laptop ko", 
     ["RAM", "SSD"], ["positive", "positive"], ["hardware", "hardware"], True),
    ("Screen brightness outdoor me kam lagti hai, colors washed out dikhte hain", 
     ["Screen brightness", "colors"], ["negative", "negative"], ["camera_display", "camera_display"], True),
    
    # Delivery & Packaging
    ("Delivery time se 2 din pehle aa gaya par packaging box torn tha", 
     ["Delivery", "packaging box"], ["positive", "negative"], ["delivery_service", "delivery_service"], True),
    ("Courier partner ne package open deliver kiya aur seller response nahi de raha", 
     ["Courier partner", "seller response"], ["negative", "negative"], ["delivery_service", "delivery_service"], True),
    ("Safe packaging ke sath genuine sealed product mila, Flipkart delivery fast thi", 
     ["packaging", "product", "delivery"], ["positive", "positive", "positive"], ["delivery_service", "product_item", "delivery_service"], True),
    ("Replacement request reject kar di customer support ne, worst experience ever", 
     ["Replacement request", "customer support"], ["negative", "negative"], ["delivery_service", "delivery_service"], True),
    
    # Fashion & Lifestyle
    ("Kapde ka material soft cotton hai lekin color wash ke baad fade ho gaya", 
     ["Kapde ka material", "color"], ["positive", "negative"], ["fabric_material", "design_appearance"], True),
    ("Kurti ki fitting ekdum accurate hai aur stitching quality strong hai", 
     ["fitting", "stitching quality"], ["positive", "positive"], ["fit_size", "quality_build"], True),
    ("Pants me pockets diye hain jo bahut useful hain, fabric breathable hai", 
     ["pockets", "fabric"], ["positive", "positive"], ["quality_build", "fabric_material"], True),
    ("Dupatta length thodi short hai par suit ka overall look royal lagta hai", 
     ["Dupatta length", "overall look"], ["negative", "positive"], ["fit_size", "design_appearance"], True),
    ("Saree ka border design gorgeous hai aur fabric lightweight hai", 
     ["border design", "fabric"], ["positive", "positive"], ["design_appearance", "fabric_material"], True),
    ("Shoes ka sole comfortable hai running ke liye but size thoda tight hai", 
     ["sole", "size"], ["positive", "negative"], ["quality_build", "fit_size"], True),
    ("Watch ka strap premium feel deta hai lekin battery 1 day me khatam ho jati hai", 
     ["strap", "battery"], ["positive", "negative"], ["quality_build", "battery_performance"], True),
    ("Jacket ka zipper first week me toot gaya, material bilkul sasta hai", 
     ["zipper", "material"], ["negative", "negative"], ["quality_build", "fabric_material"], True),
    
    # Home & Kitchen
    ("Mixer grinder ki motor powerful hai aur blades sharp hain, noise level normal hai", 
     ["motor", "blades", "noise level"], ["positive", "positive", "neutral"], ["hardware", "hardware", "hardware"], True),
    ("Induction cooktop ka heat control easy hai aur electricity consumption kam hai", 
     ["heat control", "electricity consumption"], ["positive", "positive"], ["performance", "price_value"], True),
    ("Water purifier ka filter life long hai lekin installation service late thi", 
     ["filter life", "installation service"], ["positive", "negative"], ["quality_build", "delivery_service"], True)
]

# Generate augmented dataset with variations
rows = []
for i in range(12):  # Repeat base combinations with syntactic variations
    for text, aspects, pols, cats, is_hing in AUGMENTED_ECOMMERCE_REVIEWS:
        prefix_modifiers = ["", "Overall, ", "Review: ", "My honest review - ", "Mera experience: "]
        p = random.choice(prefix_modifiers)
        full_text = p + text
        rows.append({
            "sentence": full_text,
            "aspect_terms": str(aspects),
            "at_polarity": str(pols),
            "aspect_categories": str(cats),
            "#aspect terms": len(aspects),
            "is_hinglish": is_hing,
            "source": "Flipkart_Amazon_Hinglish_Augmented"
        })

df_aug = pd.DataFrame(rows)
output_file = "data/augmented_ecommerce_hinglish_absa.csv"
df_aug.to_csv(output_file, index=False, encoding="utf-8")
print(f"Generated {len(df_aug)} augmented e-commerce Hinglish reviews -> {output_file}")
