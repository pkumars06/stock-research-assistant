"""The app's glossary: the harder words a first-time reader meets in the tabs, each in one or two plain sentences.

Written by the assistant on 2026-10-23 for people who invest through apps but never read filings. To be read
and checked by the user before the demo. "Revenue" and other everyday words are left out on purpose.
"""
GLOSSARY = [
    ("Annual report", "The yearly report a listed company must publish. It holds the accounts, the auditor's "
     "report and management's own discussion of the year."),
    ("Fiscal year (FY)", "Indian companies count their year from 1 April to 31 March. FY26 is 1 April 2025 to "
     "31 March 2026."),
    ("Crore and lakh", "1 lakh = 1,00,000 (one hundred thousand). 1 crore = 100 lakh = 1,00,00,000 (ten million). "
     "This app turns every figure into ₹ crore so the years compare, and shows the figure as printed beside it."),
    ("PDF page and printed page", "The page number in the PDF file and the number printed at the foot of the page "
     "often differ. The app gives both, for example 'PDF p151 (printed 168)'."),
    ("Standalone and consolidated", "Standalone accounts cover the company alone. Consolidated accounts add the "
     "companies it owns (its subsidiaries). This app uses standalone unless it says otherwise."),
    ("Profit before tax (PBT)", "What is left after all costs, before income tax is paid."),
    ("Profit after tax (PAT)", "Profit before tax minus income tax: the profit for the year that belongs to the owners."),
    ("Exceptional item", "A gain or a cost the company says is unusual and not part of normal trading, such as "
     "selling a building. It can make one year's profit look better or worse than the business really did."),
    ("Other income", "Money that does not come from the main business, such as interest on deposits or gains "
     "on investments."),
    ("Tax rate", "Tax divided by profit before tax. The usual company rate in India is about 25%; a rate far "
     "below it in one year often has a one-off cause."),
    ("Earnings per share (EPS)", "Profit after tax divided by the number of shares. 'Basic' EPS uses the shares "
     "that exist today."),
    ("Cash from operations (CFO)", "The cash the main business actually brought in during the year, from the "
     "cash flow statement. Unlike profit, it counts a sale only when the customer pays."),
    ("CFO ÷ PAT", "Cash from operations divided by profit after tax. Around 1 or more means the profit came in "
     "as cash; well below 1 for several years means money is stuck in unpaid bills or stock."),
    ("Borrowings", "Money owed to banks and other lenders, long-term and short-term added together."),
    ("Lease liabilities", "Rent the company has promised to pay landlords in future years, counted as a debt. "
     "Large for a restaurant company that rents its premises."),
    ("Cash and cash equivalents", "Cash in hand and in the bank, plus deposits that can be turned into cash "
     "within about three months."),
    ("Current investments", "Investments the company can sell within a year, mostly mutual funds."),
    ("Net worth (total equity)", "What the company owns minus what it owes: the owners' share of the company."),
    ("Contingent liability", "A claim against the company that it disputes, such as a tax demand it is fighting "
     "in court. It is paid only if the company loses, so it is not in the debts on the balance sheet."),
    ("Capital work in progress (CWIP)", "Money spent on things still being built, such as a new plant that has "
     "not started production."),
    ("Related party", "Someone close to the company: its directors and top managers, their relatives, and the "
     "companies they own. Deals with them have to be listed in a note to the accounts."),
    ("Key managerial personnel (KMP)", "The people who run the company: managing director, whole-time directors, "
     "chief financial officer and company secretary."),
    ("Auditor's report", "The outside accountant's opinion on whether the accounts are fair. 'Unmodified' (clean) "
     "is the normal result."),
    ("Qualified opinion", "The auditor says the accounts are fair except for a named problem. Rare, and worth "
     "reading in full when it appears."),
    ("Emphasis of matter", "The auditor points the reader to something already disclosed, without saying the "
     "accounts are wrong."),
    ("Earnings call", "A call where managers talk to analysts and investors after the results. The transcript "
     "is filed with the stock exchange."),
    ("Investor deck", "The slides a company shows investors, usually with its results. Shorter and more "
     "promotional than the annual report."),
]
