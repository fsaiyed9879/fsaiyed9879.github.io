---
layout: post
title: Deciphering Big Data
subtitle: A comprehensive reflection of the learning journey accross the 12 units of the module. It includes artefacts, summaries of activities and personal reflections related to the modules core learning outcomes.
categories: Website
tags: [big data, data management, data types, data collection and storage, data cleansing and transformation, data compliance]
---

## E-Portfolio
### Learning Outcomes

- Introduce and review various concepts of big data, technologies, and data management to enable you identify and manage challenges associated with security risks and limitations.
- Critically analyse data wrangling problems and determine appropriate methodologies and tools in problem solving.
- Explore different data types and formats. Evaluate various data storage formats ranging from structured, quasi structured, semi structured, and unstructured formats. We explore the various memory and storage requirements.
- Critically examine various data collection methods and sources. Review fact finding methods to determine the integrity, reliability and readiness of data extracted and presented for pre-processing, cleaning, and usage.
- Examine data exploration methods and analyse data for presentation in an organisation. Critically evaluate data readability, readiness, and longevity within the data Pipeline. Examine cloud services, API (Application Programming Interfaces) and how this enables data interoperability and connectivity.
- Examine and analyse the ideas and theoretical concepts underlying DBMS (Database Management Systems) Database Design and Modelling.
- Explore the future of use of data and deciphering by examining some fundamental ideas and concepts of machine learning and how these concepts are applied in various methods in handling big data.

### Unit 1 - Introduction to Big Data
At the start of the module, my understanding of big data was largely focused on scale. Unit 1 expanded this view by emphasising that big data is characterised not only by size, but also by Volume, Velocity, Variety, and Veracity (Laney, 2001). This conceptual foundation helped me understand why traditional batch-based data processing is increasingly insufficient in modern data environments.

A key theme introduced in this unit was the shift towards continuous data generation, particularly through connected devices otherwise known as Internet of things (IoT). Rather than treating data as static datasets, the module highlighted how organisations increasingly rely on real-time or near-real-time data flows to support operational and strategic decision-making. As discussed in the course material, data produced by interconnected devices can provide valuable insights across domains such as healthcare, manufacturing, and smart infrastructure, but only when it is effectively managed and interpreted (Huxley et al., 2020).

This unit also encouraged early critical reflection on the challenges that accompany such data-intensive systems. Data generated from connected environments is often inconsistent, unstructured, or incomplete, increasing the need for robust data wrangling processes before analysis can be trusted. Without appropriate cleaning and transformation, insights derived from these data streams risk being inaccurate or misleading. In addition, the scale and distributed nature of modern data collection raises significant concerns around security, privacy, and ethical use, particularly where individuals may be unaware of the extent of data being captured about them (Zuboff, 2019).

References

Huxley, C. et al. (2020) Data Wrangling in the Era of Big Data: The Role of Automation and Human Insight. [Course material].

Laney, D. (2001) 3D Data Management: Controlling Data Volume, Velocity, and Variety. META Group.

Zuboff, S. (2019) The Age of Surveillance Capitalism. New York: PublicAffairs.


### Unit 2 - Data Types and Data Sources
Unit 2 expanded my understanding of the different types of data used in big data environments, including structured, semi-structured, and unstructured data. I learned how traditional relational data differs from formats such as JSON, XML, and multimedia data, and why these distinctions matter for storage and processing decisions.

This unit highlighted how the rapid growth of unstructured data has been driven by digital platforms, social media, and connected devices. Managing such data effectively requires flexible processing frameworks and careful consideration of data quality and context when using APIs, as unstructured data is often ambiguous and noisy (Kandel et al., 2011).

Reference
Kandel, S. et al. (2011) ‘Research directions in data wrangling: Visualizations and transformations for usable and credible data’, Information Visualization, 10(4), pp. 271–288.

### Unit 3 - Data Collection and Storage
This unit focused on extracting data programmatically from multiple sources using Python and APIs, working with JSON and XML. It highlighted how APIs support scalable access and interoperability, while also requiring careful attention to schemas, documentation, and rate limits to avoid incomplete or biased data (Huxley et al., 2020). 

This week built on that by introducing web scraping: I used Requests and BeautifulSoup to search for “data scientist” on Wikipedia and saved the output in a structured format (JSON/XML). The main benefit from this exercise came as i started writing the code, changing it in the process to understand how it works.

<img width="838" height="581" alt="{CBD29B75-3C03-4E71-9204-14C7BAA9F500}" src="https://github.com/user-attachments/assets/44620660-8472-429a-842a-a8957e06f138" />

Referenes 
Huxley, C. et al. (2020) Data Wrangling in the Era of Big Data: The Role of Automation and Human Insight. [Course material].

### Unit 4 & 5 - Data Cleaning and Transformation

Unit 4 marked a turning point in the module, as it provided my first hands-on experience with cleaning and transforming datasets using Python. Activities included handling missing values, standardising data formats, and preparing raw datasets for analysis.

Although I work professionally as a data consumer, this unit gave me direct insight into the challenges faced by data engineers. I gained a strong appreciation for the importance of data preparation and how errors at this stage can negatively impact downstream analysis and modelling, a well-documented issue in data engineering practice (Rahm and Do, 2000).

Unit 5 took the UNICEF work a step further by moving from a one-off cleanup to a repeatable cleaning workflow. Rather than fixing a single DataFrame, I began thinking in terms of a pipeline that can run across many files, apply the same rules (renaming columns, handling missing data, and producing standard outputs), and leave everything ready for analysis. 

There was also a pipeline test (Unit 4) which gave me the below results:

<img width="1002" height="576" alt="{27F7179F-F8DE-416E-9A53-41A6D4BEC9F7}" src="https://github.com/user-attachments/assets/118b1245-df0c-4b13-af52-bce73a50b11b" />

<img width="1091" height="716" alt="{812584CF-03F3-4C66-A7BA-E4FA6DC2B3BF}" src="https://github.com/user-attachments/assets/04b8d937-6319-47c5-a1ce-ddea24370f21" />

Data Cleaning Exercise:
<img width="1057" height="496" alt="{5E3136FB-F1CE-429A-A0B3-7902F9DB1465}" src="https://github.com/user-attachments/assets/82af144f-c838-45c7-a16d-686188f47264" />

<img width="567" height="542" alt="{3E0133DE-8CDA-4BD5-994A-D0C739FF6E98}" src="https://github.com/user-attachments/assets/36a43de6-7f3b-4861-994d-b5c2c414178f" />
<img width="556" height="635" alt="{36E8150F-006F-4FD7-87B0-7A8475604029}" src="https://github.com/user-attachments/assets/5f2cf6cf-b566-4017-a285-0656e254ee37" />

References
Rahm, E. and Do, H.H. (2000) ‘Data Cleaning: Problems and Current Approaches’, IEEE Data Engineering Bulletin, 23(4), pp. 3–13.

### Unit 6 - Database Design and Normalisation & Unit 7 - Constructing Normalised tables and DB Build
Unit 6 focused on data storage solutions and database fundamentals. I learned about relational databases, schema design, and the principles of normalisation (Elmasri and Navathe, 2016).

These concepts later proved essential during the group project, as they provided the theoretical foundation required to design efficient and scalable data models. The unit also introduced the trade-offs between SQL and NoSQL databases, reinforcing the importance of selecting technologies based on specific use cases. 

The team report on Zerotrace is here: [DevelopmentalTeamProject_AkbarovNilssonSayied (1).pdf](https://github.com/user-attachments/files/24842418/DevelopmentalTeamProject_AkbarovNilssonSayied.1.pdf)

Unit 7 gave me 2 tasks at hand: 

Firstly, an unnormalised table to restructure into 1NF, 2NF, 3NF. While i havent actively had to draw these out in my career recently, i certainly remember doing this towards the start of my career and remember these from my studying days. Therefore, this part was quite straightforward to work on. [Normalisation Work.docx](https://github.com/user-attachments/files/24842602/Normalisation.Work.docx)


Secondly, to turn the excel sheet into a sql code which inserts the data into my GCP environment in its 3NF state.
<img width="822" height="791" alt="{DEAD3564-EA5E-4CA5-94EB-82FD1F5A6B23}" src="https://github.com/user-attachments/assets/f0182d74-0cb0-4a2a-a5e8-7a6571c4b8d5" />

### Unit 8 - Compliance
Unit 8 highlighted how strongly UK data protection law frames security as a core requirement, not an optional extra. The UK GDPR expects personal data to be handled lawfully and securely, and Article 32 in particular emphasises “appropriate” technical and organisational measures. In practical terms, that often means building in protections such as encryption and pseudonymisation, alongside controls that demonstrate ongoing risk awareness and continual improvement rather than a one-time compliance exercise (European Union, 2016; ICO, 2024).

What stood out to me is that these requirements aren’t just about preventing incidents, they also shape how systems should be designed and operated day-to-day. The emphasis on integrity and confidentiality aligns closely with the broader “privacy by design” mindset: considering security from the start, documenting decisions, and being able to justify why the controls in place are suitable for the risks involved (ICO, 2024).

References
European Union, 2016. Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 (General Data Protection Regulation). Official Journal of the European Union, L119, pp.1–88. Available at: https://gdpr-info.eu/art-32-gdpr/ [Accessed 13 July 2025].

Information Commissioner’s Office (ICO), 2024. Principle (f): Integrity and confidentiality (security). [online] ICO. Available at: https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/principles/integrity-and-confidentiality-security/ [Accessed 13 July 2025].

### Unit 9 - Database Models and Systems
This unit helped me see that DBMS choices depend on the problem, not habit. Comparing flat files, relational databases, non-relational systems, and larger platforms like data warehouses, data lakes, and Hadoop clarified the strengths and limitations of each (structure and integrity vs flexibility and scale). I also better understood how database design links to programming approaches, especially the challenges of matching object-oriented code to relational schemas. Finally, the unit reinforced that security is central across all environments—particularly in cloud and large-data setups where access control, encryption, auditing, and backups matter as much as performance.

### Unit 10 - APIs for Data Parsing
Unit 10 strengthened my understanding of how APIs enable data parsing and communication between systems, especially when working with structured formats like JSON and XML. It also highlighted that API design isn’t just about connectivity—it needs resilience and clear handling of common implementation challenges such as authentication, rate limits, error responses, and versioning. The biggest takeaway for me was the focus on security: protecting endpoints through HTTPS, access controls (API keys/tokens), input validation, and sensible permissions is essential for keeping an API reliable and preventing data exposure.

### Unit 11 - DBMS Transaction and Recovery
Unit 11 focused on how databases stay reliable when things fail. I learned how transaction processing keeps data consistent by ensuring changes are either fully committed or rolled back, supported by the ACID principles (atomicity, consistency, isolation, durability). We also looked at concurrency through interleaving/scheduling and why controls like locking help prevent issues when many transactions run at once.

On the recovery side, the unit explained how transaction logs and checkpoints support crash recovery, while serious media failures rely on backups. The Grandfather–Father–Son (GFS) method stood out as a practical backup strategy because it rotates daily, weekly, and monthly backups to balance storage use with useful restore points.

Also sumbitted an Exec Summary to build on our work from Unit 6: [Exec Summary Resubmission.docx](https://github.com/user-attachments/files/24842685/Exec.Summary.Resubmission.docx)

### Module Reflection

Working as a Data Professional in consulting has allowed me to have a good grasp of the current technologies in use for Big Data. While my knowledge of them is very technical and I am more of a data consumer, you could argue that there are many other technilogies which i havent come across in my work life and my understanding of then is more surface-level. I am aware of its volume and complexity but lack the insight into the full lifecycle of data management. This changed at the end of unit 4, where i had some first-hand experience using Python to clean and transform datasets. I had spent some time on understanding how to apply cleaning techniques e.g. handling missing values, standardising formats, gave me an appreciation for the importance of data preparation before any form of analysis or modelling. 

This module started with the basics of big data, the 4 V's of Big Data: Volume, Variety, Velocity and Veracity. How the data volume and storage has significantly increases in the last few decades and how efficiently we are able to convey the data forward. Previously we used to have the simple structured textual files, but the new world has brought with itseld the audiovisual data element and unstructured data which can be linked to the IoT. The result of this has been the growth of unstructures data in business. It is also safe to say that storing this amount of data in a secure manner has also been a challenge.

Units 2 and 3, deepened my understanding of the data extraction process. Using Python and API's to access structured and semi-structures data, exploring different file types like xml, json and csv's. 

In week 5 the unit introduced me to automation in data collection, writing a basic pyton script to see how it would work, which scraped a web page and cleaned the data for a csv export which is also something i tried in the previous module.  Seeing this process end-to-end i.e. extraction, cleaning and transformation was quite satisfying. It taught me that automation doesnt just improve efficiency but also reduces human error in repetitie data workflows.

Beyond technical proficiency, this module strengthened my confidence as both a problem soler and a collaborator. One of the most important lessons i've taken from this experience is the value of resilience in technical tasks. There were moments particularly during debugging python code or using APIs when the progress felt a little slow and daunting, but by working through these tasks, i have developed an appreciation for the language and big data as a whole. 

I have also become more aware of my own learning style, and how i manage my time for studying. While taking part in the formative activities like the group discussions was a little more time challenging for me, i was still ensuring that i did hands on experimentation to help cement difficult concepts, whether that be in the form of writing code, to connecting apis, to researching around the best practices of coding and use of other big data techncologies. 

I would confortably say that this module has been a pivot point in my academic and professional journey. While i an a data consumer in my day to day job, learning about this has given me more appreciate of the data engineers, and the troubles they have to go through for data collection, then the process of cleaning that data and making sure its useful and meaningful for the data modelling and analysis. It has definitely become a topic which interests me and has improved my skills. I plan to build on this momentum by exploring machine learning and cloud-based data pipelines, for which i plan to enrol on some of the online courses available to deepen those areas. 

To conclude this module has not only equipped me with practical toold and theorical knowledge but has also transformed the way i will approach complex technical challenges. 
