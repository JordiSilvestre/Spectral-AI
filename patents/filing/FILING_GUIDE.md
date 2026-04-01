# USPTO Non-Provisional Patent Filing Guide

**For:** SpectralAI Zero-Matrix — 3 Patent Applications
**Inventor:** Jordi Silvestre Lopez
**Organization:** LiquidBit Studio

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

For **each** of the 3 patents, prepare these files:

| Document | Format | Source |
|----------|--------|--------|
| **Specification** | PDF | Convert `patent_0X_*.md` to PDF |
| **Claims** | PDF | Included in specification (last section before Abstract) |
| **Abstract** | PDF | Included in specification (last section) |
| **Drawings** | PDF or TIFF | From illustrator (see `FIGURE_SPECS.md`) |
| **Application Data Sheet** | Web form | Based on `ADS_template.md` |
| **Inventor's Declaration** | PDF | Based on `declaration.md` (or use PTO/AIA/01 form) |
| **Micro Entity Certification** | PDF | Form SB/15a (if claiming micro entity status) |

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
- Open .md file → Ctrl+Shift+P → "Markdown PDF: Export (pdf)"

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

#### Step 1: Start New Application

1. Log in to Patent Center
2. Click **"New submission"** → **"Utility (non-provisional)"**
3. Confirm **"New application"**

#### Step 2: Application Data Sheet (ADS)

Fill in the web form using data from `ADS_template.md`:

1. **Application Information:**
   - Type: Utility, Non-Provisional
   - Title: [Use exact title from ADS template]

2. **Inventor(s):**
   - Given Name: Jordi
   - Family Name: Silvestre Lopez
   - City, State, Country: [Your details]
   - Citizenship: [Your country]

3. **Correspondence Address:**
   - [Your mailing address and email]

4. **Domestic Benefit (if any):**
   - Leave blank (no prior provisional)

5. **Related Applications:**
   - Add cross-references to the other 2 patents
   - Note: On first filing, use docket numbers; update with application numbers later

#### Step 3: Upload Documents

Upload each document with correct category:

| File | Category in Patent Center |
|------|--------------------------|
| Specification PDF | "Specification" |
| Drawing sheets | "Drawings" |
| Declaration | "Oath or Declaration" |
| Micro Entity cert. | "Petition" → "Micro Entity Certification" |

#### Step 4: Fee Payment

1. System calculates fees based on claims count
2. Verify amounts match `fee_calculation.md`
3. Pay via credit card or deposit account
4. **Save the receipt!**

#### Step 5: Submit

1. Review all uploaded documents
2. Click **"Submit"**
3. Save the **Application Number** and **Confirmation Number**
4. You'll receive an email confirmation

#### Step 6: Repeat for Patents 2 and 3

File each patent as a **separate application**. After all 3 are filed:

1. Note all 3 application numbers
2. File **supplemental ADS** for each to add the actual application numbers of the cross-referenced patents (replacing the docket numbers)

---

### Phase 3: Post-Filing

#### Immediate (within 1 week):
- [ ] Save all 3 application numbers in a secure location
- [ ] File supplemental ADS with cross-reference application numbers
- [ ] Set calendar reminders for:
  - 3 months: Check for Office Actions
  - 12 months: Maintenance review
  - 18 months: Publication check (applications publish at 18 months)

#### Ongoing:
- [ ] Respond to Office Actions within 3 months (extendable to 6 months with fee)
- [ ] File Information Disclosure Statement (IDS) if you discover new prior art
- [ ] Track application status at https://patentcenter.uspto.gov

---

## Filing Order Recommendation

**File in this order (same day if possible):**

1. **Patent 1 (RT Attention)** — Filed first because Patents 2 and 3 depend on it
2. **Patent 2 (Inception Engine)** — References Patent 1
3. **Patent 3 (Spectral Routing)** — References both Patents 1 and 2

> **Why same day?** All 3 get the same priority date, which strengthens the cross-references.

---

## Common Mistakes to Avoid

1. **Don't forget drawings** — USPTO will issue a Notice to File Missing Parts, delaying processing
2. **Don't exceed page/claim limits without paying** — System calculates fees automatically
3. **Don't forget the Declaration** — Can be filed later but delays prosecution
4. **Don't publish the paper before filing** — Once published, you have a 1-year grace period in the US, but lose most international rights. **File patents BEFORE publishing on arXiv.**
5. **Don't forget micro entity certification** — Missing it means paying 4× more

---

## Important Deadlines

| Event | Deadline | Consequence of Missing |
|-------|----------|----------------------|
| File non-provisional (if provisional filed) | 12 months from provisional date | Lose priority date |
| Respond to Office Action | 3 months (extendable to 6) | Application abandoned |
| Pay issue fee | 3 months from Notice of Allowance | Application abandoned |
| Maintenance fee (3.5 yr) | 3.5 years from grant | Patent expires |
| Maintenance fee (7.5 yr) | 7.5 years from grant | Patent expires |
| Maintenance fee (11.5 yr) | 11.5 years from grant | Patent expires |

---

## Information Disclosure Statement (IDS)

You have a duty to disclose all known prior art to USPTO. File Form SB/08 with:

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

> **Important:** File the IDS **within 3 months of filing** or **before first Office Action** to avoid additional fees.

---

## Resources

- USPTO Patent Center: https://patentcenter.uspto.gov
- Fee Schedule: https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule
- Forms: https://www.uspto.gov/patent/forms/forms-patent-applications-filed-or-after-september-16-2012
- MPEP (Manual of Patent Examining Procedure): https://www.uspto.gov/web/offices/pac/mpep/
- Pro Se Assistance: https://www.uspto.gov/patents/basics/using-legal-services/pro-se-assistance-program
