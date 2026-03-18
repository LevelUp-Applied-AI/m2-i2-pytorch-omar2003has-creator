[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field
 
 # What the model predicts:
 # Target: price_jod (The actual price of the house).
   
 # 5 Input Features:
 
 # area_sqm: The size of the house.
  
 # bedrooms: Number of rooms.
 
 # floor: Which floor the house is on.
  
 # age_years: How old the building is.
  
 # distance_to_center_km: How far it is from the city center.
 # I set up the training process with these specific settings:
 
 # Number of Epochs: 100 (The model went through the data 100 times).

 # Learning Rate: 0.01 (The speed of adjustment).
 
 # Optimizer: Adam (Handles the weight updates).
 
 # Loss Function: MSELoss (Calculates the difference between the real price and my model's guess).
 # . Training Outcome
 # Data Size: The DataFrame has 200 rows and 6 columns.

 # Loss Movement: The loss started very high at 1.95 Billion and decreased slowly throughout the 100 epochs.

 # Final Loss: The training ended with a loss of approximately 1.94 Billion.

 # Output: All results were saved in a file called predictions.csv.
 