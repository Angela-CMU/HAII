# FairPay: Empowering Fairness in Interactive Income Analysis


## Team Members
- Chien-Yu Liu (chienyul@andrew.cmu.edu)
- Yen-Ju Wu (yenjuw@andrew.cmu.edu)
- Jyoshna Sarva (jsarva@andrew.cmu.edu)


## Project Overview
Our project utilizes income datasets sourced from various Census surveys and programs, which can be found at [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). 

With this data, our aim is to uncover patterns within salary information, recognizing the paramount importance individuals place on salary in their career trajectories. We seek to identify the common factors influencing salary while scrutinizing the presence of biases within the job market. We are attentive to potential biases introduced during data collection processes and vigilant against biases emerging during data analysis, whether stemming from human factors or algorithmic/model biases. Our project not only provides users with opportunities to interact with the data and glean insights but also endeavors to identify and address potential biases throughout the entire process.


## Development
We developed our project **from scratch**, incorporating the following functionalities:

- Interactive data visualization and analysis capabilities
  - We've included histograms and pie charts, allowing users to specify the feature they're interested in.
- Model training and prediction functionalities
  - We train our models using our training data, employing both Random Forest and Logistic Regression models.
- User input prediction, allowing users to choose the preferred model for prediction
  - Users can input their data and select a model, allowing our system to make predictions based on their input.

All the functionalities mentioned above offer a highly interactive experience.


## Access our website
We've launched our website for users to explore! Feel free to visit the website we've crafted [here]().


## Building and testing locally
If you're intrigued by our project and wish to build upon it, you can clone our repository. The primary website logic resides in src/app.py and can be customized to suit your needs. To launch the website, simply navigate to the src directory and execute the command `streamlit run app.py`.


## Reflections
### Bias analysis
Throughout our project, we've identified some potential biases. 
- Firstly, there's a sampling bias, where over 90% of the data is derived from white individuals and over 66% from males. This imbalance in the data renders it unrepresentative of each demographic group, thereby limiting the generalizability of our results to the broader population. 
- Secondly, there's a concern regarding response bias, as the data is collected through survey responses. This prompts questions about the honesty of respondents and whether there were adequate data validation processes in place to ensure the reliability of the information collected.

Our main conclusion emphasizes that bias can arise not only from models but also from the process of data collection itself. Therefore, it's vital to establish guidelines for fair data collection to effectively mitigate potential biases.



