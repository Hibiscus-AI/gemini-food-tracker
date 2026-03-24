import { useState, useRef, useEffect } from 'react';

// Use relative URL in production, localhost in development
const API = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

interface NutritionInfo {
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  fiber_g: number;
}

interface IngredientNutrition {
  name: string;
  estimated_grams: number;
  nutrition: NutritionInfo;
}

interface FoodAnalysis {
  success: boolean;
  food_name?: string;
  confidence?: number;
  ingredients?: string[];
  ingredient_breakdown?: IngredientNutrition[];
  estimated_portion_g?: number;
  nutrition_per_100g?: NutritionInfo;
  nutrition_for_portion?: NutritionInfo;
  tips?: string;
  data_source?: string;
  error?: string;
}

// Common ingredient additions with nutrition per tablespoon/typical serving
const ADDABLE_INGREDIENTS: Record<string, { label: string; calories: number; protein: number; carbs: number; fat: number }> = {
  'oil': { label: 'Oil (1 tbsp)', calories: 120, protein: 0, carbs: 0, fat: 14 },
  'butter': { label: 'Butter (1 tbsp)', calories: 102, protein: 0.1, carbs: 0, fat: 11.5 },
  'cream': { label: 'Cream (2 tbsp)', calories: 100, protein: 0.8, carbs: 0.8, fat: 10 },
  'cheese': { label: 'Cheese (30g)', calories: 120, protein: 7, carbs: 0.4, fat: 10 },
  'mayo': { label: 'Mayo (1 tbsp)', calories: 94, protein: 0.1, carbs: 0.1, fat: 10 },
  'sugar': { label: 'Sugar (1 tsp)', calories: 16, protein: 0, carbs: 4, fat: 0 },
  'honey': { label: 'Honey (1 tbsp)', calories: 64, protein: 0.1, carbs: 17, fat: 0 },
  'sauce': { label: 'Sauce (2 tbsp)', calories: 30, protein: 0.5, carbs: 6, fat: 0.5 },
};

type PortionWarning = 'none' | 'zero' | 'small' | 'large';

interface CustomIngredient {
  name: string;
  calories: number;
}

function App() {
  const [step, setStep] = useState<'upload' | 'result' | 'info'>('upload');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<FoodAnalysis | null>(null);
  const [customGrams, setCustomGrams] = useState<string>('');
  const [lastValidGrams, setLastValidGrams] = useState<string>('100');
  const [portionWarning, setPortionWarning] = useState<PortionWarning>('none');

  // Ingredient management
  const [originalIngredients, setOriginalIngredients] = useState<string[]>([]);
  const [selectedIngredients, setSelectedIngredients] = useState<Set<string>>(new Set());
  const [addedIngredients, setAddedIngredients] = useState<Set<string>>(new Set());
  const [customIngredients, setCustomIngredients] = useState<CustomIngredient[]>([]);
  const [showAddIngredient, setShowAddIngredient] = useState(false);

  // Custom ingredient input
  const [customIngName, setCustomIngName] = useState('');
  const [customIngCal, setCustomIngCal] = useState('');

  // Store original nutrition for recalculation
  const [baseNutrition, setBaseNutrition] = useState<NutritionInfo | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);

  // Validate portion and set warnings
  const validatePortion = (value: string): PortionWarning => {
    const grams = parseFloat(value);
    if (isNaN(grams) || grams === 0) return 'zero';
    if (grams < 20) return 'small';
    if (grams > 1000) return 'large';
    return 'none';
  };

  // Handle portion input change
  const handlePortionChange = (value: string) => {
    // Allow empty string for typing
    if (value === '') {
      setCustomGrams('');
      setPortionWarning('zero');
      return;
    }

    const num = parseFloat(value);
    if (isNaN(num)) {
      // Reset to last valid value for non-numeric input
      setCustomGrams(lastValidGrams);
      return;
    }

    setCustomGrams(value);
    setPortionWarning(validatePortion(value));

    if (num > 0) {
      setLastValidGrams(value);
    }
  };

  // Calculate adjusted nutrition based on portion and ingredients
  const calculateNutrition = (): NutritionInfo | null => {
    if (!baseNutrition) return null;

    const grams = parseFloat(customGrams);
    if (isNaN(grams) || grams <= 0) return null;

    const multiplier = grams / 100;

    // Start with base nutrition scaled by portion
    let calories = baseNutrition.calories * multiplier;
    let protein = baseNutrition.protein_g * multiplier;
    let carbs = baseNutrition.carbs_g * multiplier;
    let fat = baseNutrition.fat_g * multiplier;
    let fiber = baseNutrition.fiber_g * multiplier;

    // Adjust for removed ingredients (reduce by ~10% per removed ingredient)
    const removedCount = originalIngredients.length - selectedIngredients.size;
    if (removedCount > 0 && originalIngredients.length > 0) {
      const reductionFactor = 1 - (removedCount * 0.1);
      calories *= Math.max(0.5, reductionFactor);
      protein *= Math.max(0.5, reductionFactor);
      carbs *= Math.max(0.5, reductionFactor);
      fat *= Math.max(0.5, reductionFactor);
      fiber *= Math.max(0.5, reductionFactor);
    }

    // Add extra ingredients from presets
    addedIngredients.forEach(ing => {
      const data = ADDABLE_INGREDIENTS[ing];
      if (data) {
        calories += data.calories;
        protein += data.protein;
        carbs += data.carbs;
        fat += data.fat;
      }
    });

    // Add custom typed ingredients
    customIngredients.forEach(ing => {
      calories += ing.calories;
    });

    return {
      calories: Math.round(calories * 10) / 10,
      protein_g: Math.round(protein * 10) / 10,
      carbs_g: Math.round(carbs * 10) / 10,
      fat_g: Math.round(fat * 10) / 10,
      fiber_g: Math.round(fiber * 10) / 10,
    };
  };

  // Auto-recalculate when inputs change
  useEffect(() => {
    if (analysis && baseNutrition) {
      const newNutrition = calculateNutrition();
      if (newNutrition) {
        setAnalysis(prev => prev ? { ...prev, nutrition_for_portion: newNutrition } : null);
      }
    }
  }, [customGrams, selectedIngredients, addedIngredients, customIngredients, baseNutrition]);

  const analyzeFood = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image');
      return;
    }

    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${API}/analyze`, { method: 'POST', body: form });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error (${res.status}): ${errorText || res.statusText}`);
      }

      const data: FoodAnalysis = await res.json();

      if (data.success) {
        setAnalysis(data);
        setCustomGrams(data.estimated_portion_g?.toString() || '100');
        setLastValidGrams(data.estimated_portion_g?.toString() || '100');
        setPortionWarning('none');

        // Store original data for recalculation
        setBaseNutrition(data.nutrition_per_100g || null);
        setOriginalIngredients(data.ingredients || []);
        setSelectedIngredients(new Set(data.ingredients || []));
        setAddedIngredients(new Set());
        setCustomIngredients([]);

        setStep('result');
      } else {
        setError(data.error || 'Analysis failed');
      }
    } catch (e) {
      if (e instanceof TypeError && e.message.includes('fetch')) {
        setError('Cannot connect to server. Make sure the backend is running on localhost:8000');
      } else if (e instanceof Error) {
        setError(e.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleIngredient = (ingredient: string) => {
    setSelectedIngredients(prev => {
      const next = new Set(prev);
      if (next.has(ingredient)) {
        next.delete(ingredient);
      } else {
        next.add(ingredient);
      }
      return next;
    });
  };

  const toggleAddedIngredient = (key: string) => {
    setAddedIngredients(prev => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const addCustomIngredient = () => {
    const name = customIngName.trim();
    const cal = parseFloat(customIngCal);

    if (!name) return;
    if (isNaN(cal) || cal < 0) return;

    setCustomIngredients(prev => [...prev, { name, calories: cal }]);
    setCustomIngName('');
    setCustomIngCal('');
  };

  const removeCustomIngredient = (index: number) => {
    setCustomIngredients(prev => prev.filter((_, i) => i !== index));
  };

  const reset = () => {
    setStep('upload');
    setPreviewUrl(null);
    setAnalysis(null);
    setCustomGrams('');
    setLastValidGrams('100');
    setPortionWarning('none');
    setError(null);
    setOriginalIngredients([]);
    setSelectedIngredients(new Set());
    setAddedIngredients(new Set());
    setCustomIngredients([]);
    setCustomIngName('');
    setCustomIngCal('');
    setBaseNutrition(null);
    setShowAddIngredient(false);
  };

  const getPortionWarningMessage = (): string | null => {
    switch (portionWarning) {
      case 'zero': return 'Please enter a valid portion size';
      case 'small': return 'Very small portion - are you sure?';
      case 'large': return 'Very large portion - are you sure?';
      default: return null;
    }
  };

  return (
    <div className="app">
      <header>
        <h1>AI Calorie Tracker</h1>
        <p className="subtitle">Powered by Gemini</p>
        <button
          className="info-btn"
          onClick={() => setStep(step === 'info' ? 'upload' : 'info')}
        >
          {step === 'info' ? '<- Back' : 'API Info'}
        </button>
      </header>

      {error && <div className="error">{error}</div>}

      <main>
        {/* Upload Step */}
        {step === 'upload' && (
          <div className="card">
            <div
              className="upload-area"
              onClick={() => fileRef.current?.click()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files[0]) analyzeFood(e.dataTransfer.files[0]);
              }}
              onDragOver={(e) => e.preventDefault()}
            >
              {loading ? (
                <div className="loading">
                  <div className="spinner" />
                  <p>Analyzing with AI...</p>
                  <small>This may take a few seconds</small>
                </div>
              ) : previewUrl ? (
                <img src={previewUrl} alt="Food" className="preview" />
              ) : (
                <>
                  <div className="icon">📸</div>
                  <h3>Upload Food Photo</h3>
                  <p>Drop image or click to browse</p>
                  <small>AI will identify food, ingredients & calories</small>
                </>
              )}
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && analyzeFood(e.target.files[0])}
              hidden
            />

            <div className="features">
              <div className="feature">Food Recognition</div>
              <div className="feature">Ingredients</div>
              <div className="feature">Nutrition</div>
            </div>
          </div>
        )}

        {/* Result Step */}
        {step === 'result' && analysis && (
          <div className="card result">
            {previewUrl && <img src={previewUrl} alt="Food" className="result-image" />}

            <div className="food-header">
              <h2>{analysis.food_name}</h2>
              <span className="confidence">
                {((analysis.confidence || 0) * 100).toFixed(0)}% confident
              </span>
              {analysis.data_source && (
                <span className={`data-source ${analysis.data_source}`}>
                  {analysis.data_source === 'verified' ? 'Verified Data' :
                   analysis.data_source === 'hybrid' ? 'DB + AI' : 'AI Estimate'}
                </span>
              )}
            </div>

            {/* Ingredient Breakdown */}
            {analysis.ingredient_breakdown && analysis.ingredient_breakdown.length > 0 && (
              <div className="ingredient-breakdown">
                <h4>Ingredient Breakdown</h4>
                <table className="breakdown-table">
                  <thead>
                    <tr>
                      <th>Ingredient</th>
                      <th>Grams</th>
                      <th>Cal</th>
                      <th>P</th>
                      <th>C</th>
                      <th>F</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.ingredient_breakdown.map((ing, i) => (
                      <tr key={i}>
                        <td>{ing.name}</td>
                        <td>{ing.estimated_grams}g</td>
                        <td>{ing.nutrition.calories.toFixed(0)}</td>
                        <td>{ing.nutrition.protein_g.toFixed(1)}</td>
                        <td>{ing.nutrition.carbs_g.toFixed(1)}</td>
                        <td>{ing.nutrition.fat_g.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Editable Ingredients */}
            <div className="ingredients">
              <h4>Ingredients (tap to toggle)</h4>
              <div className="ingredient-list">
                {originalIngredients.map((ing, i) => (
                  <button
                    key={i}
                    className={`ingredient ${selectedIngredients.has(ing) ? 'selected' : 'removed'}`}
                    onClick={() => toggleIngredient(ing)}
                  >
                    {selectedIngredients.has(ing) ? '✓' : '✗'} {ing}
                  </button>
                ))}
              </div>

              {/* Added Ingredients */}
              {(addedIngredients.size > 0 || customIngredients.length > 0) && (
                <div className="added-ingredients">
                  <small>Added:</small>
                  <div className="ingredient-list">
                    {Array.from(addedIngredients).map(key => (
                      <button
                        key={key}
                        className="ingredient added"
                        onClick={() => toggleAddedIngredient(key)}
                      >
                        ✗ {ADDABLE_INGREDIENTS[key]?.label}
                      </button>
                    ))}
                    {customIngredients.map((ing, i) => (
                      <button
                        key={`custom-${i}`}
                        className="ingredient added"
                        onClick={() => removeCustomIngredient(i)}
                      >
                        ✗ {ing.name} (+{ing.calories} cal)
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Add Ingredient Button */}
              <button
                className="add-ingredient-btn"
                onClick={() => setShowAddIngredient(!showAddIngredient)}
              >
                {showAddIngredient ? '− Close' : '+ Add Ingredient'}
              </button>

              {showAddIngredient && (
                <div className="add-ingredient-panel">
                  {/* Quick add options */}
                  <div className="quick-add-options">
                    {Object.entries(ADDABLE_INGREDIENTS).map(([key, data]) => (
                      <button
                        key={key}
                        className={`add-option ${addedIngredients.has(key) ? 'added' : ''}`}
                        onClick={() => toggleAddedIngredient(key)}
                        disabled={addedIngredients.has(key)}
                      >
                        {data.label} (+{data.calories} cal)
                      </button>
                    ))}
                  </div>

                  {/* Custom ingredient input */}
                  <div className="custom-ingredient-input">
                    <small>Or add your own:</small>
                    <div className="custom-input-row">
                      <input
                        type="text"
                        placeholder="Ingredient name"
                        value={customIngName}
                        onChange={(e) => setCustomIngName(e.target.value)}
                      />
                      <input
                        type="number"
                        placeholder="Calories"
                        value={customIngCal}
                        onChange={(e) => setCustomIngCal(e.target.value)}
                        min="0"
                      />
                      <button
                        onClick={addCustomIngredient}
                        disabled={!customIngName.trim() || !customIngCal}
                      >
                        Add
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Portion Input with Validation */}
            <div className="portion-section">
              <h4>Portion Size</h4>
              <div className="portion-input">
                <input
                  type="number"
                  value={customGrams}
                  onChange={(e) => handlePortionChange(e.target.value)}
                  placeholder="grams"
                  className={portionWarning !== 'none' ? 'warning' : ''}
                />
                <span>grams</span>
              </div>

              {/* Portion Warning */}
              {getPortionWarningMessage() && (
                <div className={`portion-warning ${portionWarning === 'zero' ? 'error' : 'warn'}`}>
                  {getPortionWarningMessage()}
                </div>
              )}

              <div className="quick-grams">
                {[50, 100, 150, 200, 250, 300].map(g => (
                  <button
                    key={g}
                    onClick={() => handlePortionChange(g.toString())}
                    className={customGrams === g.toString() ? 'active' : ''}
                  >
                    {g}g
                  </button>
                ))}
              </div>
            </div>

            {/* Nutrition Results */}
            {analysis.nutrition_for_portion && portionWarning !== 'zero' && (
              <div className="nutrition-section">
                <div className="calories-big">
                  <span className="num">{analysis.nutrition_for_portion.calories.toFixed(0)}</span>
                  <span className="label">calories</span>
                </div>

                <div className="macros">
                  <div className="macro protein">
                    <div className="val">{analysis.nutrition_for_portion.protein_g.toFixed(1)}g</div>
                    <div className="lbl">Protein</div>
                  </div>
                  <div className="macro carbs">
                    <div className="val">{analysis.nutrition_for_portion.carbs_g.toFixed(1)}g</div>
                    <div className="lbl">Carbs</div>
                  </div>
                  <div className="macro fat">
                    <div className="val">{analysis.nutrition_for_portion.fat_g.toFixed(1)}g</div>
                    <div className="lbl">Fat</div>
                  </div>
                  <div className="macro fiber">
                    <div className="val">{analysis.nutrition_for_portion.fiber_g.toFixed(1)}g</div>
                    <div className="lbl">Fiber</div>
                  </div>
                </div>

                {/* Per 100g reference */}
                {baseNutrition && (
                  <div className="per-100g">
                    <strong>Base (per 100g):</strong> {baseNutrition.calories} cal |
                    P: {baseNutrition.protein_g}g |
                    C: {baseNutrition.carbs_g}g |
                    F: {baseNutrition.fat_g}g
                  </div>
                )}

                {/* Modification indicator */}
                {(originalIngredients.length !== selectedIngredients.size || addedIngredients.size > 0 || customIngredients.length > 0) && (
                  <div className="modification-note">
                    * Adjusted for ingredient changes
                  </div>
                )}
              </div>
            )}

            {/* Tips */}
            {analysis.tips && (
              <div className="tips">
                {analysis.tips}
              </div>
            )}

            <button className="primary" onClick={reset}>Scan Another</button>
          </div>
        )}

        {/* Info Page */}
        {step === 'info' && (
          <div className="card info-page">
            <h2>Gemini 2.0 Flash API</h2>

            <div className="info-section">
              <h3>Pricing</h3>
              <table className="pricing-table">
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Cost per 1M tokens</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Input</td>
                    <td>$0.10</td>
                  </tr>
                  <tr>
                    <td>Output</td>
                    <td>$0.40</td>
                  </tr>
                </tbody>
              </table>
              <p className="info-note">
                Each food analysis costs approximately <strong>$0.0001 - $0.0002</strong> (fractions of a cent)
              </p>
            </div>

            <div className="info-section">
              <h3>Free Tier</h3>
              <ul>
                <li>15 requests per minute</li>
                <li>1,000 requests per day</li>
                <li>250,000 tokens per minute</li>
              </ul>
            </div>

            <div className="info-section">
              <h3>Model Specs</h3>
              <ul>
                <li><strong>Model:</strong> Gemini 2.0 Flash</li>
                <li><strong>Context Window:</strong> 1 million tokens</li>
                <li><strong>Capabilities:</strong> Multimodal (text + images)</li>
              </ul>
            </div>

            <div className="info-section">
              <h3>Production Estimates</h3>
              <table className="pricing-table">
                <thead>
                  <tr>
                    <th>Daily Analyses</th>
                    <th>Est. Daily Cost</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>100</td>
                    <td>~$0.02</td>
                  </tr>
                  <tr>
                    <td>1,000</td>
                    <td>~$0.20</td>
                  </tr>
                  <tr>
                    <td>10,000</td>
                    <td>~$2.00</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="info-section">
              <h3>Useful Links</h3>
              <ul>
                <li><a href="https://ai.google.dev/gemini-api/docs/pricing" target="_blank" rel="noopener noreferrer">Official Pricing Page</a></li>
                <li><a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Get API Key</a></li>
                <li><a href="https://ai.google.dev/gemini-api/docs" target="_blank" rel="noopener noreferrer">API Documentation</a></li>
              </ul>
            </div>

            <button className="primary" onClick={() => setStep('upload')}>Back to Scanner</button>
          </div>
        )}
      </main>

      <footer>
        <p>Powered by Google Gemini</p>
      </footer>
    </div>
  );
}

export default App;
