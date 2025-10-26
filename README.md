# Predicting Key Determinants of IT Project Failures in Australia
## Acknowledgements
This project was developed in collaboration with Cong Do Le, Van Hieu Tran, Khai Quang Thang at Charles Darwin University (CDU).

# Context
In the Australian context, IT and large-scale projects frequently encounter significant risks of cost-overruns, delays and outright failures. Research shows that around 18% of public sector IT initiatives become “outliers” with major cost escalations or schedule blow-outs. Factors contributing to these outcomes include underestimated complexity, insufficient requirements engineering, weak governance, and socio-technical interactions. 
This project seeks to analyse historical data on IT project outcomes and apply predictive modelling techniques to identify key determinants of project failure. The aim is to enable early-stage risk identification and informed decision-making so that inefficient projects can be avoided or right-sized before commencement.

# Objectives
* **Identify key determinants** that distinguish successful and failed IT projects using structured data and literature-based features.
* **Develop predictive models** to forecast the likelihood of project failure before initiation, supporting early risk detection and informed decision-making.

# Dataset
The dataset consists of 166 large-scale IT projects collected from 32 reliable sources, including research papers, government reports, and business websites. It contains both successful and failed projects, providing a balanced foundation for predictive analysis.

Each project is described by 21 variables covering management quality, technical competence, user involvement, planning, budget control, and resource availability. About 82.5 percent of projects are from government sectors and most have a duration of less than five years.

Data was cleaned, encoded, and standardised for modelling. Missing values, accounting for around eight percent of the dataset, were handled using mode-based imputation. The dataset supports the development of models that predict project outcomes and identify factors leading to IT project failure.

# Methodology
1. **Data Collection**

   * Compiled data on over 36,000 IT projects worldwide, including Australian government and non-government cases.
   * Selected 166 projects with complete information and labelled outcomes (“Successful” or “Failed”).
   * Defined 21 project features based on insights from 46 research papers and 14 industry reports, covering areas such as management quality, planning, technical competence, user involvement, and resource availability.

<img width="972" height="525" alt="image" src="https://github.com/user-attachments/assets/f1a8ce5a-3c12-4292-88d9-a006b488701f" />


2. **Data Processing**

   * Extracted and cleaned both structured and unstructured data using Python tools such as BeautifulSoup and Regular Expressions, and Power Automate Web Scraper.
   * Encoded categorical features and balanced the dataset (83 successful and 83 failed projects) for fair model training.

3. **Exploratory Data Analysis (EDA)**

   * Analysed distributions of project duration, cost, and sector to identify trends.
   * Found that most projects lasted under five years, with 82.5% originating from government sectors.

4. **Model Development**

   * Applied several machine learning algorithms including Naive Bayes, Random Forest, Support Vector Machine (SVM), and XGBoost.
   * Evaluated models using accuracy, precision, recall, and F1-score metrics.
   * Determined the most influential features leading to project failure: user involvement, technical competence, planning quality, and resource adequacy.

5. **Results and Validation**

   * Models achieved high predictive accuracy, with SVM reaching up to 94%.
   * Results confirmed that project outcomes can be reliably predicted using early-stage project characteristics, enabling better strategic and funding decisions.

<img width="934" height="526" alt="image" src="https://github.com/user-attachments/assets/3c4cbb9e-72ce-4a77-a77f-d830a10705a2" />


# Key findings

1. Analysis of 166 IT projects found that nearly half failed to meet objectives due to weak management, unrealistic expectations, and limited stakeholder engagement. Failures were most common in government-led projects, where complex governance processes contributed to inefficiency.

2. Among the 21 assessed features, the most influential determinants of project outcomes were user involvement, technical competence, planning quality, resource availability, and stability of requirements. Projects with high user participation and technically capable teams were significantly more successful.

3. Poor planning and unstable requirements were recurring issues in failed projects, often leading to cost and time overruns. Frequent changes to scope and unclear requirements reduced delivery efficiency and increased risk.

4. The Random Forest and Support Vector Machine (SVM) models achieved the best predictive performance, with accuracy reaching 94 percent. These models were able to predict project outcomes reliably based on early-stage characteristics.

5. Exploratory data analysis showed that 95 percent of projects were completed within five years, indicating that longer project durations tend to increase failure risk. Extended timelines also correlated with higher rates of cost escalation and scope variation.

6. Projects with low user satisfaction and poor technical literacy among stakeholders were more likely to fail, highlighting the importance of communication and end-user understanding.

7. Successful projects consistently demonstrated strong executive management, proper planning, and clearly defined requirements, underscoring the value of effective leadership and governance.

# Recommendations

1. Strengthen project planning by defining clear, stable, and comprehensive requirements before implementation begins.
2. Increase stakeholder and user involvement across all project phases to improve alignment with actual needs and promote ownership of outcomes.
3. Invest in building technical competence among project teams and improve technical literacy for stakeholders to enhance decision-making and implementation quality.
4. Establish realistic budgets and schedules to prevent over-commitment and reduce the likelihood of time or cost overruns.
5. Implement structured change-control mechanisms to manage requirement adjustments and maintain project stability.
6. Ensure adequate allocation of financial, human, and technological resources before commencement to prevent project disruptions.
7. Strengthen executive oversight and IT governance frameworks to maintain accountability and clear direction.
8. Apply predictive analytics to evaluate project feasibility and risk before initiation, prioritising projects with higher success probability.
9. Adopt shorter, modular project structures to reduce exposure to uncertainty and evolving external factors.
10. Conduct post-project reviews to capture lessons learned and refine best practices for future IT project planning and execution.

# Limitation
<img width="956" height="545" alt="image" src="https://github.com/user-attachments/assets/05cbf502-ab15-4194-baeb-6f3511534b64" />

# References

1. Chen, H.L. (2015). *Performance measurement and the prediction of capital project failure*. *International Journal of Project Management*, 33(6), 1393–1404. [https://doi.org/10.1016/j.ijproman.2015.02.009](https://doi.org/10.1016/j.ijproman.2015.02.009)

2. Dwivedi, Y.K., Wastell, D., Laumer, S., Henriksen, H.Z., Myers, M.D., Bunker, D., Elbanna, A., Ravishankar, M.N. and Srivastava, S.C. (2015). *Research on information systems failures and successes: Status update and future directions*. *Information Systems Frontiers*, 17(1), 143–157. [https://doi.org/10.1007/s10796-014-9500-y](https://doi.org/10.1007/s10796-014-9500-y)

3. Owolabi, H.A., Bilal, M., Oyedele, L.O., Alaka, H.A., Ajayi, S.O. and Akinade, O.O. (2020). *Predicting completion risk in PPP projects using big data analytics*. *IEEE Transactions on Engineering Management*, 67(2), 430–442. [https://doi.org/10.1109/TEM.2018.2876321](https://doi.org/10.1109/TEM.2018.2876321)

4. Public Accounts Committee. (2014). *Management of ICT Projects by Government Agencies*. Legislative Assembly of the Northern Territory, Darwin, Australia. ISBN 9780987432872.

5. Standing, C., Guilfoyle, A., Lin, C. and Love, P.E.D. (2006). *The attribution of success and failure in IT projects*. *Industrial Management & Data Systems*, 106(8), 1148–1165. [https://doi.org/10.1108/02635570610710809](https://doi.org/10.1108/02635570610710809)
