# Precise Constraint Examples

This document shows examples of how to use precise row-level conditional constraints in Matrix AI.

## How It Works

1. **User describes data** with precise constraints
2. **Gemini generates** schema and seed data (approximate distribution)
3. **SDV scales up** the data to target size
4. **AI extracts precise constraints** from original description
5. **AI enforces exact counts/percentages** by modifying synthetic data

---

## Example 1: Exact Count with Condition

**User Input:**
```
Generate 1000 users with subscription data. Exactly 100 users should have 'no_churn'
status, and these should be users with subscription_type='premium'.
```

**What Happens:**
- Gemini generates schema with users table (subscription_type, churn_status columns)
- SDV generates 1000 users
- AI detects constraint: "exactly 100 users with subscription_type='premium' should have churn='no'"
- AI enforces: Randomly selects 100 premium users and sets their churn_status to 'no_churn'

**Result:** Exactly 100 premium users with no churn ✅

---

## Example 2: Percentage-Based Distribution

**User Input:**
```
Create 5000 orders. 25% of orders should have status='cancelled',
and 60% should be status='completed'.
```

**What Happens:**
- Gemini generates orders table with status column
- SDV generates 5000 orders
- AI detects constraints:
  - "25% of orders should be status='cancelled'" → 1250 orders
  - "60% of orders should be status='completed'" → 3000 orders
- AI enforces: Randomly sets status for exact counts

**Result:** Exactly 1250 cancelled, 3000 completed ✅

---

## Example 3: Conditional Distribution

**User Input:**
```
Generate 2000 subscriptions. For users with premium_type='gold',
80% should have renewal='yes'. For silver users, only 40% renew.
```

**What Happens:**
- Gemini generates subscriptions with premium_type and renewal columns
- SDV generates 2000 subscriptions
- AI detects constraints:
  - "premium_type='gold' → 80% renewal='yes'"
  - "premium_type='silver' → 40% renewal='yes'"
- AI enforces: For each group, sets exact percentages

**Result:** Precise renewal rates per subscription tier ✅

---

## Example 4: Date-Based Conditional

**User Input:**
```
Create 3000 customer records. Customers who joined before 2023-01-01
should have exactly 30% churn rate. Newer customers have 10% churn.
```

**What Happens:**
- Gemini generates customers with join_date and churn columns
- SDV generates 3000 customers
- AI detects constraints:
  - "join_date < 2023-01-01 → 30% churn"
  - "join_date >= 2023-01-01 → 10% churn"
- AI enforces: Calculates counts per group and sets exact percentages

**Result:** Precise churn rates based on cohort ✅

---

## Example 5: Multiple Conditions

**User Input:**
```
Generate 10000 transactions. Exactly 500 transactions should be:
- amount > 1000 AND
- status = 'flagged' AND
- customer_type = 'new'
```

**What Happens:**
- Gemini generates transactions with amount, status, customer_type columns
- SDV generates 10000 transactions
- AI detects complex constraint with multiple conditions
- AI enforces: Finds/creates 500 rows matching all conditions

**Result:** Exactly 500 high-value flagged transactions from new customers ✅

---

## Supported Constraint Types

✅ **Exact counts:** "exactly 100", "precisely 50 out of 1000"
✅ **Percentages:** "30% of records", "half of the users"
✅ **Conditional:** "if X then Y", "when status='active' then churn='no'"
✅ **Multiple conditions:** "X AND Y", combined filters
✅ **Date/time conditions:** "before 2023", "after subscription_start"
✅ **Numerical comparisons:** ">1000", "between 18 and 65"

---

## Tips for Best Results

1. **Be explicit:** "exactly 100 users" is better than "some users"
2. **Specify conditions clearly:** Use column=value format
3. **Use percentages or exact counts:** Both work well
4. **Multiple constraints:** List them clearly in your description
5. **Test with samples:** Review the data to ensure constraints were applied

---

## Technical Implementation

The system uses a 2-stage approach:

**Stage 1: Statistical Learning (SDV)**
- SDV learns general patterns from seed data
- Generates realistic distributions

**Stage 2: Precise Enforcement (AI)**
- Gemini extracts exact quantitative constraints
- Python code applies precise modifications
- Ensures exact counts/percentages are met

This hybrid approach gives you both **realistic data** AND **precise control**.
