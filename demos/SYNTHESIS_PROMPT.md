# How the demo documents were made

All demo documents are synthetic (fictional companies). They were LLM-written with this prompt, then lightly edited:

> Write realistic but fictional internal documents for a shipping company. Set 1: three short HR policies for
> shore staff (leave; business travel & expenses; hybrid/remote work), each 250-400 words with concrete numbers,
> deadlines and named systems. Set 2: a ship spares catalogue CSV (~14 parts with part_no, description, equipment,
> maker, unit_price_usd, stock_qty, lead_time_days, supplier) and the supply terms of the two suppliers in the CSV
> (minimum order, discounts, delivery, payment, warranty, returns) - make the two suppliers' terms differ.
> Do NOT include salary bands or payroll dates in the HR set (used to test the groundability guard).
