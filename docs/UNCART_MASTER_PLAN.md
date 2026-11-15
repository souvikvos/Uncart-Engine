# UNCart — Developer Roadmap & Source of Truth
**Project ID:** ML_29 — Customer Product Return Prediction  
**Core Goal:** A clean dashboard for e-commerce sellers to flag high-risk returns before shipping.

---

## 1. What Is UNCart? (In 2 Sentences)
UNCart is a web dashboard for e-commerce store owners. When new orders come in, it predicts which orders have a high risk of being returned, shows the exact reasons why (using Explainable AI), and gives an AI recommendation to prevent shipping losses.

---

## 2. The 4 Screens We Are Building

```
[ 1. Orders Table ] ──> View incoming orders with Risk Badges (Green / Yellow / Red) + CSV Uploader
        │
        ├──> [ 2. Risk Inspector Modal ] (Click any row) ──> SHAP bars (Why it's risky) + Gemini AI advice
        │
        ├──> [ 3. Live Order Simulator ] ──> Interactive form for professor to test custom orders live
        │
        └──> [ 4. Analytics & Benchmark ] ──> Category return rate charts + LR vs. Random Forest metrics
```

1. **Screen 1: Orders Table (Main Page):**
   * A clean table showing: `Order ID`, `Category`, `Price`, `Payment`, and `Risk Level` (Low / Medium / High).
   * A button to **"Upload CSV"** to screen 50+ orders at once.
2. **Screen 2: Risk Inspector Modal (Popup on Click):**
   * A popup when clicking any order showing:
     * **SHAP Bar Chart:** Why it's risky (e.g., `+30% Cash on Delivery`, `+25% Apparel`, `-10% Loyal Customer`).
     * **AI Advice Box:** A 2-sentence Gemini recommendation based on store policy.
3. **Screen 3: Live Order Simulator (The Viva Form):**
   * Simple inputs for Category, Price, Payment Method, and Quantity.
   * Click **"Predict Risk"** $\rightarrow$ Instantly calculates and animates the probability and reasons on the spot.
4. **Screen 4: Analytics & Model Evaluation Tab:**
   * **Recharts Bar Chart:** Return rate across categories (Apparel ~32%, Footwear ~25%, Electronics ~8%).
   * **Academic Benchmark Table:** Side-by-side comparison of **Logistic Regression (ML_29)** vs. **Random Forest** (Accuracy, Precision, Recall, ROC-AUC).

---

## 3. Team Responsibilities (Consistent & Simple)

| Member | Area | What They Actually Build | Viva Defense |
| :--- | :--- | :--- | :--- |
| **Souvik (Lead)** | **AI / ML & Engine** | Runs the Logistic Regression training script, sets up the SHAP explanation logic, and wires the Gemini AI advice. | Machine Learning math (ML_29), SHAP explainability, and AI recommendations. |
| **Soyam** | **Backend & DB** | Sets up the Express.js server, connects MongoDB, creates the 4 simple API routes, and handles the CSV upload. | Express.js API design, MongoDB models, and CSV data processing. |
| **Srayan** | **Frontend UI** | Builds the React frontend (Orders table, Simulator form, and Recharts graphs) using Tailwind CSS. | UI/UX design, component structure, and data charts. |
| **Sourav** | **QA & Docs** | Creates 3 sample CSV files (`normal`, `high_risk`, `test`), tests the web pages, and formats the college report/PPT. | Testing methodology, edge cases, and final project documentation. |

---

## 4. How the System Works (The Simple Flow)

```
[ Srayan's React UI ]
        │  (Sends Order JSON)
        ▼
[ Soyam's Express Server + MongoDB ]
        │  (Forwards to Python)
        ▼
[ Souvik's Python Script (FastAPI) ]
   ├── 1. Logistic Regression calculates probability (e.g. 78%)
   ├── 2. SHAP calculates the reasons (e.g. COD + Apparel)
   └── 3. Gemini generates the 2-sentence advice
```

---

## 5. The 4 Express API Routes (For Soyam)

1. `POST /api/predict` — Takes order JSON, calls Souvik's Python script, and returns risk score + SHAP reasons.
2. `POST /api/upload-csv` — Takes an uploaded CSV file, saves orders to MongoDB, and scores them.
3. `GET /api/orders` — Fetches saved orders from MongoDB to display in Srayan's table.
4. `GET /api/analytics` — Calculates simple totals (total orders, high-risk count, return rates).

---

## 6. The Order Data Fields (Simple Table)

| Field Name | Example Value | Why It Matters |
| :--- | :--- | :--- |
| `product_category` | Apparel / Footwear / Electronics | Apparel has high return rates (~30%); electronics are low. |
| `product_price` | ₹1,500 | High-priced items carry higher financial risk. |
| `payment_method` | Cash on Delivery / UPI / Card | COD has 3x higher doorstep rejection risk. |
| `item_quantity` | 1, 2, or 3 | Ordering 2+ of the same dress indicates size-trialing ("wardrobing"). |
| `customer_past_return_rate` | 0.25 (25%) | Does this customer return things often? |
| `discount_percent` | 20% | High discounts correlate with impulse buying. |
| **`is_returned`** (Target) | 0 (Kept) or 1 (Returned) | What the model predicts. |

---

## 7. Viva Cheat Sheet (Only 4 Points to Remember)

1. **Who is this for?** E-commerce store owners. Returned orders cost sellers ₹250–₹350 in courier fees. UNCart catches risky orders *before* shipping.
2. **What model is used?** **Logistic Regression** (Mandatory Project ML_29). It takes order features and squashes them into a 0% to 100% probability using a Sigmoid curve.
3. **Why did we benchmark Random Forest?** To prove academic rigor. Logistic Regression achieved ~82% ROC-AUC with full transparency, while Random Forest achieved ~84% but acted as a black box.
4. **What is SHAP?** A math technique that explains the prediction: showing exactly how much COD or Apparel pushed the risk score up.

---

## 8. Step-by-Step Build Checklist

* [ ] **Step 1 (Souvik):** Run `train.py` to train the Logistic Regression model and save `uncart_pipeline.joblib`.
* [ ] **Step 2 (Souvik):** Run the Python engine (`engine.py`) to expose the `/predict` endpoint.
* [ ] **Step 3 (Soyam):** Create the Express app with the 4 routes and connect MongoDB.
* [ ] **Step 4 (Srayan):** Build the React UI (Dashboard Table, Inspector Modal, Simulator Form).
* [ ] **Step 5 (Sourav):** Put together the test CSV files and format the presentation slides.
