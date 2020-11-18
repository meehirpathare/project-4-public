
## **Metis Project 4 (NLP): Text Generation and Sentiment Analysis of College Confidential**

### **Goal**
For this project, I hoped to build a text generator built on a corpus of College Confidential forum posts. Ideally, the generated text will have a level of semantic sophistication 
and be able to respond in human readable sentences. I hope to be able to indentify posts and classify then according to a few post "archetypes" and respond accordingly. 

### **Background**
College Confidential is a college admissions forum founded in 2001. It is well known for its *"What are my chances?"* posts for elite universities.
Responses to these posts are often (overly) critical. This presented an intriguing dataset due to its *passionate* nature and somewhat-structured format.

### **Methodology**
* Webscraped over 16,000 comments from approximately 200 threads in the University of Pennsylvania forum
* NLP pre-processing using an NLPPipe object: removed caps, punctuation, digits, usernames
* Added stop words to increase data seperability
* Explored data and visualized with PCA and TSNE
* Trained sequential keras LSTM model on corpus of comment text with five epochs (over 2,000,000 parameters) run on Google Cloud instance
* Generated text based on this model

### **Results**
Topic modelling of the data did not seperate the corpus in a readily apparent way. The narrow focus of the corpus as well as a number of synonyms and acronyms lessened the 
effectiveness of techniques. 

### **Future Work**
* fit the data using a model that takes into account semantics. 
* use domain knowledge to find and group synonyms
* tune neural nets and allow for a longer run time

### **Tools Used**
* Pandas
* Jupyter
* BeautifulSoup
* Keras
* Google Cloud Platform
* LSTM

### **Skills**
* Neural Nets
* Unsupervised learning
* Web Scraping
* Natural Language processing
