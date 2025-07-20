import numpy as np
import pandas as pd

# Test case 1: Mixed object array
print("Test case 1: Mixed object array")
try:
    a = np.array(['a', 'b', np.nan], dtype=object)
    print(f"Array: {a}, dtype: {a.dtype}")
    print(f"np.isnan(a): {np.isnan(a)}")
except Exception as e:
    print(f"Error: {e}")

# Test case 2: Convert to float first
print("\nTest case 2: Convert to float first")
try:
    a = np.array(['a', 'b', np.nan], dtype=object)
    print(f"Array before: {a}, dtype: {a.dtype}")
    # Try to convert to float
    try:
        a_float = a.astype(float)
        print(f"Array after conversion: {a_float}, dtype: {a_float.dtype}")
        print(f"np.isnan(a_float): {np.isnan(a_float)}")
    except Exception as e:
        print(f"Conversion error: {e}")
        
    # Alternative: only check numeric values
    mask = pd.Series(a).apply(lambda x: isinstance(x, (int, float)))
    numeric_only = np.array([x if mask.iloc[i] else 0 for i, x in enumerate(a)])
    print(f"Numeric only array: {numeric_only}")
    print(f"Safe isnan check: {pd.isna(a)}")
except Exception as e:
    print(f"Error: {e}")

# Test case 3: From pandas dataframe
print("\nTest case 3: From pandas dataframe")
try:
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': ['a', 'b', 'c']
    })
    print(f"DataFrame:\n{df}")
    print(f"df.dtypes:\n{df.dtypes}")
    
    # Convert to numpy array
    arr = df.values
    print(f"Array from DataFrame: {arr}, dtype: {arr.dtype}")
    
    # Try isnan
    try:
        print(f"np.isnan(arr): {np.isnan(arr)}")
    except Exception as e:
        print(f"isnan error: {e}")
    
    # Convert to float first
    try:
        arr_float = arr.astype(float)
        print(f"This will fail because strings can't be converted to float")
    except Exception as e:
        print(f"Conversion error: {e}")
    
    # Safe approach
    print(f"Safe approach with pd.isna():\n{pd.isna(df)}")
except Exception as e:
    print(f"Error: {e}")

# Test case 4: Recommended solution
print("\nTest case 4: Recommended solution")
try:
    # Create features with mixed types
    features = np.array([
        [1, 'a', 2.5],
        [3, 'b', np.nan],
        [np.nan, 'c', 4.5]
    ], dtype=object)
    print(f"Features array:\n{features}")
    
    # Convert only numeric columns to float
    numeric_mask = np.zeros(features.shape, dtype=bool)
    for i in range(features.shape[1]):
        try:
            # Check if column can be converted to float
            col = features[:, i].astype(float)
            numeric_mask[:, i] = True
        except:
            pass
    
    print(f"Numeric mask:\n{numeric_mask}")
    
    # Create a new array with NaNs replaced with 0 in numeric columns
    cleaned_features = features.copy()
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            if numeric_mask[i, j]:
                val = features[i, j]
                if pd.isna(val):
                    cleaned_features[i, j] = 0.0
    
    print(f"Cleaned features:\n{cleaned_features}")
except Exception as e:
    print(f"Error: {e}")

print("\nConclusion:")
print("1. np.isnan() doesn't work reliably on object arrays with mixed types")
print("2. Converting mixed type arrays to float will fail")
print("3. pd.isna() is safer for checking NaN values in mixed type data")
print("4. For numeric operations, separate numeric and non-numeric data") 