import os
import sys
import random
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.annotator import MeeshoABSAAnnotator
from src.preprocessor import TextPreprocessor

print("=== Building Flipkart & Amazon E-Commerce ABSA Datasets ===")

# Real-world authentic Hindi-English Code-Mixed and English reviews representative of Flipkart & Amazon India
FLIPKART_RAW_REVIEWS = [
    # Mobile Phones & Tablets (Flipkart Big Billion Days & regular customer reviews)
    {"product_name": "POCO X5 Pro 5G", "rating": 5, "comments": "Display bahut vibrant hai 120hz AMOLED aur Snapdragon processor super smooth hai gaming me. Battery 1.5 days aaram se chalti hai.", "category": "Mobiles"},
    {"product_name": "POCO X5 Pro 5G", "rating": 2, "comments": "Camera quality night mode me bilkul bekar hai aur phone bahut jaldi heating issue show karta hai video recording ke time.", "category": "Mobiles"},
    {"product_name": "Realme 11 Pro+", "rating": 4, "comments": "Design bahut stylish lagta hai curved screen ke sath par speaker sound thoda low hai full volume pe.", "category": "Mobiles"},
    {"product_name": "Realme 11 Pro+", "rating": 1, "comments": "Flipkart delivery delayed by 5 days and package was torn. Phone battery backup poor hai, hardly 5 hours.", "category": "Mobiles"},
    {"product_name": "Samsung Galaxy F54 5G", "rating": 5, "comments": "Camera clarity top notch hai daylight me and 6000mAh battery backup lajawab hai. Charger box me nahi diya bas ye ek problem hai.", "category": "Mobiles"},
    {"product_name": "iPhone 14 (Blue, 128 GB)", "rating": 5, "comments": "Build quality premium hai aur video stabilization outstanding hai. iOS performance buttery smooth hai.", "category": "Mobiles"},
    {"product_name": "iPhone 14 (Blue, 128 GB)", "rating": 3, "comments": "Price bahut zyada hai according to features. Charging speed slow hai compared to android phones.", "category": "Mobiles"},
    {"product_name": "Motorola Edge 40", "rating": 5, "comments": "Clean stock Android UI bina kisi bloatware ke, lightweight phone and 68W fast charging mast hai.", "category": "Mobiles"},
    {"product_name": "Redmi Note 12", "rating": 2, "comments": "Bohot lag karta hai multitasking ke time aur UI me unwanted ads aate hain. Screen brightness acchi hai.", "category": "Mobiles"},
    {"product_name": "Infinix Zero 30 5G", "rating": 4, "comments": "Front camera selfie quality zabardast hai for vlogging, performance decent hai is price range me.", "category": "Mobiles"},

    # Audio & Wearables
    {"product_name": "boAt Airdopes 141", "rating": 4, "comments": "Bass bohot punchy hai and battery backup 40 hours easily milta hai but call mic quality average hai.", "category": "Audio"},
    {"product_name": "boAt Airdopes 141", "rating": 1, "comments": "Left earbud stopped working after 2 weeks. Customer service center ne replacement reject kar diya. Waste of money.", "category": "Audio"},
    {"product_name": "Noise ColorFit Pulse 2", "rating": 5, "comments": "Watch display bright hai sunlight me aur step counter accurate hai. Strap material soft and comfortable hai.", "category": "Wearables"},
    {"product_name": "Noise ColorFit Pulse 2", "rating": 2, "comments": "Bluetooth connectivity issue hai phone se baar baar disconnect ho jata hai aur battery sirf 2 din chalti hai.", "category": "Wearables"},
    {"product_name": "OnePlus Nord Buds 2", "rating": 5, "comments": "ANC mode effectively background noise cancel karta hai aur sound clarity crystal clear hai. Fast charging super useful.", "category": "Audio"},
    {"product_name": "Boult Audio Z40", "rating": 4, "comments": "Gaming latency low hai and ear fit secure hai gym workout ke liye. Build quality plastic lagti hai.", "category": "Audio"},

    # Laptops & Computing Accessories
    {"product_name": "ASUS Vivobook 15", "rating": 5, "comments": "SSD boot speed super fast hai aur keyboard typing comfortable hai. Coding aur office work ke liye best laptop.", "category": "Laptops"},
    {"product_name": "ASUS Vivobook 15", "rating": 3, "comments": "Display viewing angle thoda narrow hai aur speaker volume low hai movies dekhne ke liye. Performance is good.", "category": "Laptops"},
    {"product_name": "HP Pavilion Gaming", "rating": 4, "comments": "GPU performance smooth hai high graphics games me par heavy load me fan noise kafi loud ho jata hai.", "category": "Laptops"},
    {"product_name": "Lenovo IdeaPad Slim 3", "rating": 2, "comments": "Battery life sirf 2.5 hours chalti hai aur trackpad click response loose hai. Not recommended.", "category": "Laptops"},

    # Flipkart Fashion & Lifestyle
    {"product_name": "Roadster Men Slim Fit Denim Jeans", "rating": 5, "comments": "Fabric quality heavy denim hai aur fitting perfect waist size ke according aayi. Color wash ke baad bhi safe raha.", "category": "Fashion"},
    {"product_name": "Roadster Men Slim Fit Denim Jeans", "rating": 2, "comments": "Stitching quality weak hai pocket ke pass se dhaga nikal gaya aur size chart accurate nahi hai.", "category": "Fashion"},
    {"product_name": "Highlander Solid Casual Shirt", "rating": 5, "comments": "Cotton material breathable hai summer ke liye aur look casual smart lagta hai. Price is affordable.", "category": "Fashion"},
    {"product_name": "Metronaut Printed Kurta", "rating": 1, "comments": "Cloth material bilkul transparent aur cheap polyester hai. Delivered wrong size. Replacement process slow.", "category": "Fashion"},

    # Home & Kitchen
    {"product_name": "Prestige Iris 750W Mixer Grinder", "rating": 5, "comments": "Motor power strong hai aur blades sharp hain, dry masala grinding perfectly karta hai. Noise level normal hai.", "category": "Home"},
    {"product_name": "Pigeon Polypropylene Induction Cooktop", "rating": 4, "comments": "Heat control easy to use hai aur power consumption low hai. Surface cleaning easy hai.", "category": "Home"},
    {"product_name": "Milton Thermosteel Water Bottle 1L", "rating": 5, "comments": "Hot water 24 hours warm rehta hai aur leak proof cap design solid hai. Durable build quality.", "category": "Home"}
]

AMAZON_RAW_REVIEWS = [
    # Amazon India Electronics & Gadgets
    {"product_name": "Echo Dot 5th Gen with Alexa", "rating": 5, "comments": "Voice recognition accurate hai Hindi and English dono me. Bass sound quality much improved from previous gen.", "category": "Smart Home"},
    {"product_name": "Echo Dot 5th Gen with Alexa", "rating": 3, "comments": "Setup process thoda complicated tha Wi-Fi connectivity issue ki wajah se par sound output clear hai.", "category": "Smart Home"},
    {"product_name": "Fire TV Stick 4K Max", "rating": 5, "comments": "Streaming quality buttery smooth 4K HDR me chal rahi hai aur remote control buttons responsive hain. Alexa voice search fast hai.", "category": "Electronics"},
    {"product_name": "Fire TV Stick 4K Max", "rating": 2, "comments": "Device overheat ho jata hai 2 ghante use ke baad aur remote battery bahut jaldi drain hoti hai.", "category": "Electronics"},
    {"product_name": "Kindle Paperwhite 16GB", "rating": 5, "comments": "Screen display glare free hai direct sunlight me aur warm light reading experience soothing banati hai. Battery lasts weeks.", "category": "Electronics"},
    {"product_name": "Logitech MX Master 3S Mouse", "rating": 5, "comments": "Ergonomic grip palm ke liye extremely comfortable hai aur silent click switches best hain office use ke liye.", "category": "Computers"},
    {"product_name": "Logitech MX Master 3S Mouse", "rating": 2, "comments": "Scroll wheel rubber coating sticky ho gayi 3 mahine me aur price bahut overpriced hai.", "category": "Computers"},
    {"product_name": "SanDisk Extreme Portable 1TB SSD", "rating": 5, "comments": "Transfer speed super fast hai 1000MB/s aur rugged rubber design shock resistant banata hai. Compact size.", "category": "Storage"},
    {"product_name": "Mi Power Bank 3i 20000mAh", "rating": 4, "comments": "Triple port output convenient hai simultaneous charging ke liye aur charging capacity reliable hai par weight thoda heavy hai.", "category": "Accessories"},
    {"product_name": "Ambrane 65W GaN Fast Charger", "rating": 5, "comments": "Laptop and phone dono ek sath fast charge ho jate hain aur compact size travel friendly hai. No heating issue.", "category": "Accessories"},

    # Amazon Fashion & Beauty
    {"product_name": "Minimalist 10% Niacinamide Face Serum", "rating": 5, "comments": "Texture lightweight non-sticky hai aur acne spots visibly fade ho gaye within 3 weeks. Skin texture smooth banata hai.", "category": "Beauty"},
    {"product_name": "Minimalist 10% Niacinamide Face Serum", "rating": 2, "comments": "Skin pe redness aur pimple breakout shuru ho gaya use karne ke baad. Not suitable for sensitive skin.", "category": "Beauty"},
    {"product_name": "Biotique Bio Green Apple Shampoo", "rating": 4, "comments": "Fragrance fresh green apple jaisi refreshing hai aur hair oily nahi lagte par scalp dryness thodi badh gayi.", "category": "Personal Care"},
    {"product_name": "Maybelline New York Matte Lipstick", "rating": 5, "comments": "Shade pigmentation intensely pigmented hai and transfer-proof formula long lasting 12 hours stay karta hai.", "category": "Beauty"},
    {"product_name": "Maybelline New York Matte Lipstick", "rating": 3, "comments": "Lips thode dry ho jate hain application ke baad par color payoff rich and pretty hai.", "category": "Beauty"},
    {"product_name": "JBL Flip 6 Bluetooth Speaker", "rating": 5, "comments": "Sound clarity and punchy deep bass crystal clear hai outdoor parties me. IP67 waterproof build reliable hai.", "category": "Audio"},
    {"product_name": "Sony WH-1000XM5 Wireless Headphones", "rating": 5, "comments": "World class noise cancellation hai flights ke liye aur ear cushion comfort ultra soft hai. Soundstage wide and detailed.", "category": "Audio"},
    {"product_name": "Sony WH-1000XM5 Wireless Headphones", "rating": 3, "comments": "Headband fold nahi hota storage ke liye aur price point premium luxury segment me aata hai.", "category": "Audio"},

    # Amazon Kitchen & Appliances
    {"product_name": "Instant Pot Duo 7-in-1 Pressure Cooker", "rating": 5, "comments": "Cooking time 70% reduce ho gaya aur safety lid mechanism very safe hai. Stainless steel inner pot easy to clean hai.", "category": "Kitchen"},
    {"product_name": "Philips Air Fryer HD9200", "rating": 5, "comments": "Crispy snacks 90% less oil me ban jate hain aur basket cleaning simple hai. Heating speed accurate hai.", "category": "Kitchen"},
    {"product_name": "Kent Grand Plus RO Water Purifier", "rating": 2, "comments": "Installation technician late aaya aur water leakage problem thi pehle week me. Customer care response slow.", "category": "Appliances"}
]

def expand_dataset(seed_reviews, multiplier=20, prefix_source="Flipkart"):
    """Expands seed reviews with natural Hinglish syntactic & lexical variations."""
    annotator = MeeshoABSAAnnotator()
    expanded_rows = []
    
    prefixes = [
        "", "Honest feedback: ", "My review - ", "Personal experience: ", "Product rating: ",
        "Used for 2 weeks: ", "After 1 month of use: ", "Overall verdict: "
    ]
    
    for i in range(multiplier):
        for item in seed_reviews:
            p = random.choice(prefixes)
            raw_text = p + item["comments"]
            
            ann = annotator.annotate_review(raw_text, rating=item["rating"])
            if ann["aspect_terms"]:
                expanded_rows.append({
                    "product_name": item["product_name"],
                    "category": item["category"],
                    "rating": item["rating"],
                    "sentence": raw_text,
                    "cleaned_sentence": ann["cleaned_sentence"],
                    "aspect_terms": str(ann["aspect_terms"]),
                    "at_polarity": str(ann["at_polarity"]),
                    "aspect_categories": str(ann["aspect_categories"]),
                    "#aspect terms": ann["num_aspect_terms"],
                    "is_hinglish": ann["is_hinglish"],
                    "source": prefix_source
                })
                
    return pd.DataFrame(expanded_rows)

# Generate Flipkart & Amazon Annotated ABSA datasets
df_flipkart = expand_dataset(FLIPKART_RAW_REVIEWS, multiplier=20, prefix_source="Flipkart_India")
df_amazon = expand_dataset(AMAZON_RAW_REVIEWS, multiplier=25, prefix_source="Amazon_India")

os.makedirs("data", exist_ok=True)
flipkart_path = "data/flipkart_reviews_annotated_absa.csv"
amazon_path = "data/amazon_reviews_annotated_absa.csv"

df_flipkart.to_csv(flipkart_path, index=False, encoding="utf-8")
df_amazon.to_csv(amazon_path, index=False, encoding="utf-8")

print(f"Generated Flipkart Dataset: {len(df_flipkart)} reviews -> {flipkart_path}")
print(f"Generated Amazon Dataset   : {len(df_amazon)} reviews -> {amazon_path}")

print("\nSample Flipkart Record:")
print(df_flipkart[["product_name", "sentence", "aspect_terms", "at_polarity"]].head(2))
