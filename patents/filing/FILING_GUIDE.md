# USPTO Provisional Patent Filing Guide

**For:** SpectralAI Zero-Matrix — 3 Patent Applications
**Inventor:** Jordi Silvestre Lopez
**Filing as:** Individual inventor (pro se)
**Strategy:** File provisionals now ($195 total), convert to non-provisional within 12 months (~$5,440)

---

## Prerequisites (Complete Before Filing)

- [ ] **USPTO.gov account** — Create at https://patentcenter.uspto.gov
- [ ] **Customer Number** — Obtained during registration
- [ ] **Digital Certificate** — Required for electronic filing (or use Patent Center's built-in auth)
- [ ] **Entity Status determined** — Micro, Small, or Large (see `fee_calculation.md`)
- [ ] **Figures completed** — All 17 figures (6+5+6) in PNG/PDF format (see `FIGURE_SPECS.md`)
- [ ] **Payment method ready** — Credit card or USPTO deposit account

---

## Step-by-Step Filing Process

### Phase 1: Prepare Documents (Before Going Online)

Provisional applications require **fewer documents** than non-provisional. For **each** of the 3 patents, prepare:

| Document | Format | Required? | Source |
|----------|--------|-----------|--------|
| **Specification** | PDF | YES | Convert `patent_0X_*.md` to PDF |
| **Drawings** | PDF or TIFF | YES | From illustrator (see `FIGURE_SPECS.md`) |
| **Cover Sheet** | Web form | YES | Patent Center web form (or Form SB/16) |
| **Micro Entity Certification** | PDF | YES (if micro) | Form SB/15a |

**NOT needed for provisional:**
- ~~Application Data Sheet (ADS)~~ — the cover sheet replaces it
- ~~Inventor's Declaration~~ — not required for provisional
- ~~Formal claims examination~~ — claims are included in the specification but are not examined at this stage

> **Note on claims:** Keep the claims in your specification PDF as-is. They establish the scope of your invention and will be examined when you convert to non-provisional. They just don't need to be formally formatted or counted for fee purposes at the provisional stage.

#### Converting Markdown to PDF

Option 1 — **Pandoc** (recommended):
```bash
pandoc patents/patent_01_rt_attention.md -o patent_01.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm \
  -V fontsize=12pt
```

Option 2 — **VS Code** with Markdown PDF extension:
- Install "Markdown PDF" extension
- Open .md file -> Ctrl+Shift+P -> "Markdown PDF: Export (pdf)"

Option 3 — **Online converter:**
- https://www.markdowntopdf.com/
- Upload .md, download .pdf

> **Important:** Ensure the PDF has:
> - Minimum 2.5 cm (1 inch) margins on all sides
> - 12pt or larger font
> - Sequential page numbers
> - Line numbering (optional but helpful for examiner)

---

### Phase 2: File on USPTO Patent Center

**URL:** https://patentcenter.uspto.gov

#### Step 1: Start New Provisional Application

1. Log in to Patent Center
2. Click **"New submission"** -> **"Provisional"**
3. Confirm **"New application"**

#### Step 2: Cover Sheet

The provisional cover sheet is much simpler than the full ADS. Fill in:

1. **Title of Invention:** Use exact title from the patent specification
2. **Inventor:**
   - Given Name: Jordi
   - Family Name: Silvestre Lopez
   - City, State/Province, Country: [Your details]
3. **Correspondence Address:**
   - [Your mailing address and email]

That is all. No related applications, no domestic benefit, no assignee information needed at this stage.

#### Step 3: Upload Documents

Upload each document with correct category:

| File | Category in Patent Center |
|------|--------------------------|
| Specification PDF | "Specification" |
| Drawing sheets | "Drawings" |
| Micro Entity cert. | "Petition" -> "Micro Entity Certification" |

> **That is it.** No Declaration, no formal claims document separate from the spec.

#### Step 4: Fee Payment

Provisional application filing fees:

| Entity Status | Fee per Patent | Total (3 Patents) |
|--------------|----------------|-------------------|
| **Micro Entity** | **$65** | **$195** |
| Small Entity | $130 | $390 |
| Large Entity | $320 | $960 |

1. Verify the system shows the correct fee
2. Pay via credit card or deposit account
3. **Save the receipt**

#### Step 5: Submit and Save Confirmation

1. Review all uploaded documents
2. Click **"Submit"**
3. **Save the Application Number** — format will be 6X/XXX,XXX
4. **Save the Confirmation Number**
5. You will receive an email confirmation
6. Print or save the filing receipt PDF

#### Step 6: Repeat for Patents 2 and 3

File each patent as a **separate provisional application**.

> **IMPORTANT: File all 3 on the same day.** This gives all 3 patents the same priority date, which strengthens the cross-references and simplifies the timeline.

After all 3 are filed:
1. Record all 3 application numbers together in a secure location
2. Note the priority date (the filing date — today)

---

### Phase 3: Post-Provisional Actions

#### Immediate (within 1 week):

- [ ] Save all 3 application numbers and confirmation numbers in a secure location
- [ ] Back up all filed PDFs (specification + drawings) — these are the documents that define your priority date

#### Now Safe to Do:

- [ ] **Publish paper on arXiv** — your priority date is secured, publication cannot invalidate your patents
- [ ] **Present at conferences** — same reasoning
- [ ] **Contact potential sponsors or licensees** — share freely, priority is locked in
- [ ] **Post about the research publicly** — blog, social media, etc.

#### Calendar Reminders (SET THESE NOW):

| When | Action | Why |
|------|--------|-----|
| **Month 3** (July 2026) | Review provisional — any improvements to add? | Can file continuation-in-part if needed |
| **Month 6** (October 2026) | Start drafting non-provisional versions | Plenty of time to polish claims |
| **Month 9** (January 2027) | **START preparing non-provisional filings** | Need time for formal claims, ADS, Declaration |
| **Month 11** (March 2027) | **DEADLINE WARNING** — non-provisional must be filed soon | One month left before losing everything |
| **Month 12** (April 2, 2027) | **ABSOLUTE DEADLINE** — file non-provisional or lose priority | Miss this = lose provisional fee + priority date |

---

### Phase 4: Convert to Non-Provisional (Month 9-10)

When you are ready to convert (ideally by month 9-10 after filing):

#### What You Need for Non-Provisional

| Document | Notes |
|----------|-------|
| **Full Application Data Sheet (ADS)** | Now required — use `ADS_template.md` |
| **Inventor's Declaration** | Now required — Form PTO/AIA/01 or `declaration.md` |
| **Specification PDF** | Updated version if improvements were made |
| **Drawings PDF** | Updated if needed |
| **Micro Entity Certification** | Form SB/15a (if still qualifying) |
| **Formal Claims** | Now examined and counted for fees |

#### Filing the Non-Provisional

1. On Patent Center: **"New submission"** -> **"Utility (non-provisional)"**
2. In the ADS, under **"Domestic Benefit / National Stage Information"**:
   - Select **"Claims benefit of provisional application"**
   - Enter the provisional Application Number (6X/XXX,XXX)
   - Enter the provisional Filing Date
3. Repeat for each of the 3 patents, referencing the correct provisional
4. **Cross-reference** the other 2 patents in the "Related Applications" section

#### Non-Provisional Fees (Micro Entity)

| Fee | Per Patent | Total (3 Patents) |
|-----|-----------|-------------------|
| Basic filing fee | $80 | $240 |
| Search fee | $165 | $495 |
| Examination fee | $190 | $570 |
| Claims (3 independent, 20 total est.) | ~$0 | ~$0 |
| **Subtotal per patent** | **~$435** | **~$1,305** |

> Note: Additional fees apply if you exceed 3 independent claims or 20 total claims per patent. Excess independent claims cost $230 each, excess dependent claims cost $42 each (micro entity rates).

> **Total estimated non-provisional cost:** ~$1,305 (micro entity) for basic filing of all 3. Additional costs may apply for excess claims, extensions, and issue fees later.

---

## Fee Summary

| Phase | Cost (Micro Entity) | When |
|-------|-------------------|------|
| **Provisional filing (3 patents)** | **$195** | **Now** |
| Non-provisional conversion (3 patents) | ~$1,305 | Month 9-10 |
| Issue fees (if granted, 3 patents) | ~$600 | After allowance |
| **Total through grant** | **~$2,100** | Over 2-3 years |

---

## Filing Order

**File in this order (ALL on the same day):**

1. **Patent 1 (RT Attention)** — Filed first because Patents 2 and 3 depend on it
2. **Patent 2 (Inception Engine)** — References Patent 1
3. **Patent 3 (Spectral Routing)** — References both Patents 1 and 2

> **Why same day?** All 3 get the same priority date, which strengthens the cross-references and gives you a single 12-month deadline to track.

---

## Common Mistakes to Avoid

1. **Don't publish on arXiv BEFORE filing the provisional** — File first, publish after. Even one day matters for establishing priority.
2. **Don't forget the 12-month deadline** — If you miss the deadline to convert to non-provisional, you lose the priority date AND the $195 provisional filing fee. The provisional simply expires with no patent protection.
3. **Don't forget drawings** — USPTO will issue a Notice to File Missing Parts, delaying processing
4. **Don't forget micro entity certification** — Missing it means paying 4x more
5. **Don't assume provisional = patent** — A provisional application is NOT a patent. It only secures a priority date. You MUST file the non-provisional within 12 months.
6. **Don't modify the specification after filing** — The provisional specification defines what your priority date covers. New matter added in the non-provisional does NOT get the provisional's priority date.
7. **Don't lose the application numbers** — You need them to claim benefit in the non-provisional filing

---

## Important Deadlines

| Event | Deadline | Consequence of Missing |
|-------|----------|----------------------|
| **Convert provisional to non-provisional** | **12 months from provisional filing** | **Lose priority date + provisional fees wasted** |
| Respond to Office Action (non-prov) | 3 months (extendable to 6) | Application abandoned |
| Pay issue fee | 3 months from Notice of Allowance | Application abandoned |
| Maintenance fee (3.5 yr) | 3.5 years from grant | Patent expires |
| Maintenance fee (7.5 yr) | 7.5 years from grant | Patent expires |
| Maintenance fee (11.5 yr) | 11.5 years from grant | Patent expires |

---

## Information Disclosure Statement (IDS)

You have a duty to disclose all known prior art to USPTO. File Form SB/08 with the non-provisional (not required for provisional, but good to prepare now).

### Known Prior Art (include in IDS):

**US Patents/Applications:**
- None directly known (novel area — RT Cores for ML routing)

**Foreign Documents:**
- None known

**Non-Patent Literature:**
1. Vaswani et al., "Attention Is All You Need," NeurIPS 2017
2. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention," NeurIPS 2022
3. Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models," JMLR 2022
4. Jiang et al., "Mixtral of Experts," arXiv:2401.04088, 2024
5. Muennighoff et al., "OLMoE: Open Mixture-of-Experts Language Models," arXiv:2409.02060, 2024
6. NVIDIA, "OptiX Programming Guide," Version 8.0+
7. Karras, "Maximizing Parallelism in the Construction of BVHs," HPG 2012
8. Meneses et al., "RT Cores for Scientific Computing: A Survey," arXiv:2603.28771, 2026
9. Your own arXiv paper (if published before examination) — file supplemental IDS

> **Important:** File the IDS **within 3 months of the non-provisional filing** or **before first Office Action** to avoid additional fees.

---

## Resources

- USPTO Patent Center: https://patentcenter.uspto.gov
- Fee Schedule: https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule
- Forms: https://www.uspto.gov/patent/forms/forms-patent-applications-filed-or-after-september-16-2012
- MPEP (Manual of Patent Examining Procedure): https://www.uspto.gov/web/offices/pac/mpep/
- Pro Se Assistance: https://www.uspto.gov/patents/basics/using-legal-services/pro-se-assistance-program
- Provisional Application FAQ: https://www.uspto.gov/patents/basics/types-patent-applications/provisional-application-patent
