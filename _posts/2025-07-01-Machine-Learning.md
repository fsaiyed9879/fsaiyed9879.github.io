---
layout: post
title: Machine Learning
subtitle: A comprehensive reflection of the learning journey accross the 12 units of the module. It includes artefacts, summaries of activities and personal reflections related to the modules core learning outcomes.
categories: Website
tags: [machine learning, python, exploratory data analysis, correlations, regression]
---

**Learning Outcomes for the Machine Learning Unit:**

    1- Articulate the legal, social, ethical, and professional issues faced by machine learning professionals.
    2- Understand the applicability and challenges associated with different datasets for the use of machine learning algorithms.
    3- Apply and critically appraise machine learning techniques to real-world problems, particularly where technical risk and uncertainty is involved.
    4- Systematically develop and implement the skills required to be effective member of a development team in a virtual professional environment, adopting real-life perspectives on team roles and organisation.

**Unit 1**
Machine Learning is truly everywhere these days, driving decisions from our online browsing to big financial choices, all thanks to the explosion of real-time data that helps these systems learn and get smarter day by day, while also increating ethical issues and spread of misinformation (Schwab, 2016). This past 12 weeks, I have really delved deep into the world od ML, tracing its journey from foundational ideas (Bishop, 2006) and core statistical principles (Crawford, 2006) right up to the cutting-edge applications like deep learning in social computing (Zhao & Li, 2018). What really struck me is both the exciting oppornity ML brings to the data world but also the serious challenges that come with relying on such algorithms. I explored how big data, ML, and AI all fit together in today's vast data landscape. Now, i have a much clearer picture of ML's huge potential to reshape industries (Metcalf; 2024; WEF; 2025), the specific skills needed to master this field and crucially the ethical pitfalls we must discuss (BBC News, 2020).

References:
BBC News. (2020) 'Facebook sued over Cambridge Analytica data scandal', BBC News, 16 March. Available at: [https://www.bbc.co.uk/news/technology-54722362] (Accessed: 20 October 2025).

Bishop, C. (2006) Pattern Recognition and Machine Learning. New York, NY: Springer.

Crawford, S. L. (2006) 'Correlation and Regression', Circulation, 114(9), pp. 2083–2088.

Metcalf, G.S. (2024) 'An Introduction to Industry 5.0: History, Foundations, and Futures', In: Nousala, S., Metcalf, G., Ing, D. (eds) Industry 4.0 to Industry 5.0. Translational Systems Sciences, vol 41. Singapore: Springer.

Schwab, K. (2016) The Fourth Industrial Revolution. Geneva: World Economic Forum

World Economic Forum (WEF) (2025) The Future of Jobs Report 2025. Available at: https://www.weforum.org/publications/the-future-of-jobs-report-2025/ (Accessed: 20 October 2025).

Zhao, X. and Li, C. (2018) 'Deep Learning in Social Computing', In: Deng, L., Liu, Y. (edns) Deep Learning in Natural Language Processing. Singapore: Springer, pp. 255-288.

**Unit 2 - Exploratory Data Analysis**
This unit introduced me with EDA practically, where I performed data analysis on the Auto-MPD dataset provided online in a csv format. I uploaded this data into google colab to identify data completeness and estimate skewness & kurtosis. I plotted a bunch of visuals to present back results of my EDA. It proved a great strating point especially given the simplicity of the dataset I was using, allowing me experiment. 
<img width="608" height="503" alt="Unit 2Data Exploration" src="https://github.com/user-attachments/assets/0b863e7d-a1ed-4c0f-8bd6-c692ca5b3866" />
<img width="1090" height="520" alt="unit 2 skewness" src="https://github.com/user-attachments/assets/4b7fba1b-6a16-4b7a-bb5d-3b4adf98b003" />
<img width="1086" height="524" alt="unit 2 kurtosis" src="https://github.com/user-attachments/assets/511e2551-fbe8-41bb-9cbd-d92ede74562b" />
<img width="475" height="701" alt="unit 2 correlations" src="https://github.com/user-attachments/assets/f244568a-14a4-449c-8adc-dfa0888b4ce9" />
<img width="704" height="611" alt="unit 2 correlation heatmap" src="https://github.com/user-attachments/assets/7d0f3d1d-a271-4f49-9235-27a3ba7ba8bb" />

**Unit 3 & 4 - Correlation and Regression**
The unit had predefined python scripts with visualisation showing me examples of correlation and different types of regression. The introduction of polunomial, linear and pearson formed a solid understanding to the concept. Also, to be able to modify the set parameters helped gain better understanding of its affect on the different types of regression and correlation. 

A couple of Examples below:
<img width="500" height="252" alt="polynomial" src="https://github.com/user-attachments/assets/35a98db3-8bea-46ff-b3d0-013619d3ad1b" />
<img width="552" height="413" alt="pearsons" src="https://github.com/user-attachments/assets/d5c5e3fd-819c-4318-8d3c-df8b7fac5758" />

**Unit 5 - Clustering**
Learnt about Clustering in detail, what they are, the different types etc. Lead on to learning the Jaccard Coefficient Calculations.

The Definition of Jaccard correlation is: <img width="165" height="55" alt="image" src="https://github.com/user-attachments/assets/e06e06b6-cf9d-4df5-9292-1408ec64174e" />

Dataset:
<img width="852" height="183" alt="{0DAB8BE0-88DF-4F2B-A734-C94857CBB73E}" src="https://github.com/user-attachments/assets/f9cccbc0-1283-46fc-9420-f201ce2c595f" />

Calculating the Jaccard Coefficient of the 3 pairs below:

First Calculation - Pair (Jack, Mary)
Attributes: Fever, Cough, Test-1, Test-2, Test-3, Test-4
Jack: Y, N, P, N, N, A
Mary: Y, N, P, A, P, N

Matches: Fever (Y,Y), Cough (N,N), Test-1 (P,P) = 3 matches
Total attributes compared: 6

J(Jack,Mary) = 3/6 = 0.5

Second Calculation - Pair (Jack, Jim)
Jack: Y, N, P, N, N, A
Jim: Y, P, N, N, N, A

Matches: Fever (Y,Y), Test-2 (N,N), Test-3 (N,N), Test-4 (A,A) = 4 matches
Total attributes compared: 6

J(Jack,Jim) = 4/6 = 0.67

Third Calculation - Pair (Jim, Mary)
Jim: Y, P, N, N, N, A
Mary: Y, N, P, A, P, N

Matches: Fever (Y,Y) = 1 match
Total attributes compared: 6

J(Jim,Mary) = 1/6 = 0.17

**Unit 6 - Development Team Project**
We were required to work as a team of 4 for a development team project, where the task demanded us to complete an analysis report and submit a peer review of our team members. It proved to be a bit of a struggle to come together for open discussions at a time which suited everyone. However, we overcame this by being able to communicate via teams with the other team members, arranging workshops where whoever was available could join in to work on the team project and their assigned duties. While we knew that ML can take some time to train models and present back results, this task certainly highlighted the patience needed to train a model. 

Team project can be found here: [Identifying Competitive Market Segments and Pricing Recommendations for Airbnb Listings in NYC.docx](https://github.com/user-attachments/files/22996558/Identifying.Competitive.Market.Segments.and.Pricing.Recommendations.for.Airbnb.Listings.in.NYC.docx)


**Unit 7 & 8 - Artificial Neural Networks**
These units has perceptron activities and grafient cost function. 

In the perceptron activity, I applied my understanding of the Perceptron by experimenting with different input values in the code provided and weight adjustments to trigger neuron activation. However, I observed that if the inputs remain constant and only the weights are alteres, the neuron fails to activate.

The gradient cost function was focused on finding the lowest possible cost with the fewest iterations in the code. The exercise involced achieving that target by comparing 2 scenarios: the initial setup used 100 iterations with a learning rate of 0.08 while the modified setup reduced the process to 60 iterations with a learning rate of 0.07.

Unit 9 and 10 - 


