# AAI6610_Fall2025
course project at Northeastern University in AAI 6610, Fall2025

To run this project:

1) Install the dependencies:
```bash
pip install -r requirements.txt

```
2) Change the file name from .env.example to .env and replace all variables with its keys respectively.Run the primary script (gui.py).
```bash
python .pipeline1/main_control/gui.py

```
3) You will see a Graphical User Interface.

![Screenshot of UI](image.png)

- Select the crawler scripts and click on Selected Scrappers Button
 
- After collecting the data, it can be executed in steps (step two) or directly in one click (step three).
 

4) The files crawled by the crawler will be stored in "scrapers\outputs".

Semantic filtering is located in clustering\outputs\filter_stats_all_sources and clustering\outputs\filtered_posts_all_sources.
The clustering text results will be located in pipeline1\clustering\outputs\cluster_output, the pattern of txt is like:

======================================================================
Cluster 0 | Size: 80 documents
======================================================================
Keywords: forecasting, time, prediction, series, time series, bayesian, conformal, deep, source, probabilistic

Summary: title denoising esg quantifying missing machine prediction interval author sergio caprioli jacopo foschi riccardo crupi alessandro sabatino published 29t14 44z arxiv 2407 20047 category relevance score abstract environmental social governance esg datasets frequently plagued significant gap leading inconsistency esg rating due varying imputation method explores application established machine technique imputing missing real world esg dataset emphasizing quantification prediction interval employing multiple imputation strategy study assesses robustness imputation method quantifies associated missing finding highlight importance probabilistic machine providing better understanding esg score thereby addressing inherent risk wrong rating due incomplete approach improves imputation practice enhance reliability esg rating title enhancing communication time series prediction insight recommendation author apoorva karagappa pawandeep kaur betz jonas gilg moritz zeumer andreas gerndt published 22t13 55z arxiv 2408 12365 category relevance score abstract world increasingly relies mathematical forecast different area effective communication time series prediction important informed decision making study explores user estimate probabilistic time series prediction different variant line chart depicting examines role individual characteristic influence user reported metric estimation addressing aspect aim enhance understanding visualization improving communication time series forecast visualization design prediction dashboard world increasingly relies mathematical forecast different area effective communication time series prediction important informed decision making study explores user estimate probabilistic time series prediction different variant line chart depicting examines role individual characteristic influence user reported metric estimation addressing aspect aim enhance understanding visualization improving communication time series forecast visualization design prediction dashboard title conformal quantification electricity price prediction risk averse storage arbitrage author saud alghumayjan ming bolun published 10t00 15z arxiv 2412 07075 category math stat relevance score abstract proposes risk averse approach energy storage price arbitrage leveraging conformal quantification electricity price prediction method address significant challenge posed inherent volatility real time electricity price create substantial risk financial loss energy storage participant relying future price forecast plan operation framework comprises two layer prediction quantify real time price confidence interval high coverage framework distribution free work underlying point prediction evaluate quantification effectiveness storage price arbitrage application managing risk participating real time market design risk averse policy profit maximization energy storage arbitrage find safest storage schedule minimal loss using historical new york state synthetic price prediction evaluation demonstrate framework achieve good profit margin less purchase title augmented contrastive clustering aware prototyping time series test time adaptation author peiliang gong mohamed ragab min zhenghua chen yongyi published 01t11 17z arxiv 2501 01472 category relevance score abstract test time adaptation aim adapt pre trained deep neural network using solely online unlabelled test inference although tta shown promise visual application potential time series context remains largely unexplored existing tta method originally designed visual task may effectively handle complex temporal dynamic real world time series resulting suboptimal adaptation performance address gap propose augmented contrastive clustering aware prototyping accup straightforward yet effective tta method time series initially approach employ augmentation ensemble time series capture diverse temporal information variation incorporating aware prototype distill essential characteristic additionally introduce entropy comparison scheme selectively acquire confident prediction enhancing reliability pseudo label furthermore utilize augmented contrastive clustering enhance feature discriminability mitigate error accumulation noisy pseudo label promoting cohesive clustering within class facilitating clear separation different class extensive experiment conducted three real world time series datasets additional visual dataset demonstrate effectiveness generalization potential proposed method advancing underexplored realm tta time series title mamba time series forecasting quantification author pedro pessoa paul campitelli douglas shepherd banu ozkan steve press published 13t20 38z arxiv 2503 10873 category stat nlin relevance score abstract state space mamba recently garnered attention time series forecasting due ability capture sequence pattern however electricity consumption benchmark mamba forecast exhibit mean error approximately similarly traffic occupancy benchmark mean error reach discrepancy leaf wonder whether prediction simply inaccurate fall within error given spread historical address limitation propose method quantify predictive mamba forecast propose dual network framework based mamba architecture probabilistic forecasting one network generates point forecast estimate predictive modeling variance abbreviate tool mamba probabilistic time series forecasting mamba probtsf code implementation available github evaluating approach synthetic real world benchmark datasets find kullback leibler divergence learned distribution limit infinite converge zero correctly capture underlying probability distribution reduced order synthetic real world benchmark demonstrating effectiveness find electricity consumption traffic occupancy benchmark true trajectory stay within predicted interval two sigma level time end consideration potential limitation adjustment improve performance consideration applying framework process purely largely stochastic dynamic stochastic change accumulate observed example pure brownian motion molecular dynamic trajectory

Representative: [arxiv-papers]_2407.20047.txt
Title: Title: Denoising ESG: quantifying data uncertainty from missing data with Machine Learning and prediction intervals

Preview:
Title: Denoising ESG: quantifying data uncertainty from missing data with Machine Learning and prediction intervals Authors: Sergio Caprioli, Jacopo Foschi, Riccardo Crupi, Alessandro Sabatino arXiv ID: 2407.20047 Categories: cs.LG Relevance Score: 3.0

======================================================================