"""
Helper Script: Extract Feature Names from Trained Model
This creates model_features.csv if you don't have it
"""

import pickle
import pandas as pd

print("="*60)
print("EXTRACTING FEATURE NAMES FROM MODEL")
print("="*60)

# Load the model
try:
    with open('random_forest_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
    
    # Try to get feature names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
        print(f"✓ Found {len(feature_names)} features")
        
        # Save to CSV
        df = pd.DataFrame({'Feature': feature_names})
        df.to_csv('model_features.csv', index=False)
        print("✓ Saved to 'model_features.csv'")
        
        print("\nFirst 10 features:")
        for i, feat in enumerate(feature_names[:10], 1):
            print(f"   {i}. {feat}")
        
    else:
        print("⚠️  Model doesn't have feature_names_in_ attribute")
        print("   This is okay - the Streamlit app will work without it")
        
except FileNotFoundError:
    print("❌ Error: 'best_random_forest_model.pkl' not found")
    print("   Make sure the model file is in the same directory")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("DONE!")
print("="*60)