---
layout: post
title: Stock Market Clustering with Prim's Algorithm
comments: True
permalink: stock-clustering
---

This analysis explores clustering in the S&P 500 index. By identifying stocks that move together, we can predict stock price movements of similar stocks. Through the pandas, numpy, and matplotlib libraries, we calculate metrics to model the stock market. By employing Prim and Kruskal's clustering algorithms, we can discover patterns in the underlying structure of equities.

To do this, we use the 2015 daily adjusted close prices of all of the firms listed on the S&P 500. The data can be found on and downloaded from websites such as Google Finance or Yahoo Finance. The exact dataset used in this post can be found [here](https://www.dropbox.com/s/ra5a4w1h82fjq6u/SP_500_close_2015.csv) and [here](https://www.dropbox.com/s/ahof61wjlaz27ee/SP_500_firms.csv).

We will be using Jupyter notebooks (an application to run annotated, interactive python code) to explore the data and conduct our analysis.

First we import the Python data science libraries (Pandas, Numpy, MatPlotLib) as well as the csv module so we can import the dataset.


```python
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline
```

First, we import the csv data file into a Pandas DataFrame. The first thing we must do when faced with a new dataset is to examine what is inside. In this case, the columns are each company in the S&P 500 and each row presents the company's closing price for that day.


```python
priceData = pd.read_csv('SP_500_close_2015.csv',index_col = 0)
priceData.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MMM</th>
      <th>ABT</th>
      <th>ABBV</th>
      <th>ACN</th>
      <th>ATVI</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>156.678596</td>
      <td>43.160459</td>
      <td>61.986410</td>
      <td>86.129228</td>
      <td>19.765196</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>153.145069</td>
      <td>43.170070</td>
      <td>60.819874</td>
      <td>84.674997</td>
      <td>19.490271</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>151.511999</td>
      <td>42.679830</td>
      <td>60.518833</td>
      <td>84.064223</td>
      <td>19.126976</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>152.610267</td>
      <td>43.025880</td>
      <td>62.964797</td>
      <td>85.828689</td>
      <td>18.714587</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>156.267949</td>
      <td>43.910238</td>
      <td>63.623323</td>
      <td>87.137495</td>
      <td>18.901144</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 496 columns</p>
</div>



Many people know that AAPL represents Apple's stock ticker, or that GOOG is Google stock (now Alphabet, Google's holding company). But
unless a person is intimately familiar with the stock market, they may not know what ABT or ZTS represents.

We introduce another dataset that links a company's ticker symbol with their Name and Industry Sector. 


```python
firms = pd.read_csv("SP_500_firms.csv")
firms.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Name</th>
      <th>Sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M Company</td>
      <td>Industrials</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABT</td>
      <td>Abbott Laboratories</td>
      <td>Health Care</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>Health Care</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACN</td>
      <td>Accenture plc</td>
      <td>Information Technology</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ATVI</td>
      <td>Activision Blizzard</td>
      <td>Information Technology</td>
    </tr>
  </tbody>
</table>
</div>



## Stock Returns

The first thing we wish to do is standardize our price data. Each stock trades at a different price, and companies can control share prices through stock split. Therefore, price data on its own does not tell us much regarding a stock's performance. 

Investors prefer a \$1 increase in a stock originally worth \$10, rather than the same \$1 increase in a stock originally worth \$100. Instead of examining price differences, we look at percent change from one day to another to determine returns to initial investment.


```python
percent_change = priceData.pct_change()
percent_change = percent_change.drop(percent_change.index[0])
percent_change.head()

#Or equivalently without using Pandas' built-in 
#percent change function.
percent_changeD = {}
for i in percent_change:
    percent_changeD[i] = []
    for j in range(1,(len(priceData))):
        ret = (priceData[i][j]-priceData[i][j-1])/priceData[i][j-1]
        percent_changeD[i].append(ret)
        
percent_change2 = pd.DataFrame(data = percent_changeD, 
			index=priceData.index[1:])
```

### Maximum / Minimum Daily Returns

After calculating each stock's daily return, we wish to see which stocks experienced the best and worst days over the course of the year.


```python
def fullname(ts):
    return firms[firms.Symbol == ts].Name.values[0]

currMax = 0
for i in percent_change2:
    for j in percent_change2.index:
        if percent_change2[i][j] > currMax:
            currMax = percent_change2[i][j]
            bestCo = i
            bestDate = j
            
print (fullname(bestCo), bestDate, currMax)
```

    Freeport-McMoran Cp & Gld 2015-08-27 0.286616201466


Freeport-McMoran CP&Gld (FCX) experienced the highest daily return of any stock in the S&P 500 during 2015. It yielded 28.66% on August 27, 2015 as its adjusted close price rose from 7.92 to 10.19. 

A cursory Google search reveals that on that date, FXC announced that it was lowering spending by cutting 10% of its workforce and production in response to plunging copper prices. In addition, on that same date, activist investor Carl Icahn disclosed an 8.5 percent stake in the company. A more in-depth look at the company's historical trading data reveals that on this date, the number of shares sold was about 2.5 times the stock's average trading volume.

Despite the impressive daily gain, the company still lost much of its value over the course of the year, with shares falling from \$23.36 to \$6.77 (-71% return) due to steep drops in commodity prices.

We can find the worst day any stock had in 2015 using a similar method.


```python
currMin = 1
for i in percent_change2:
    for j in percent_change2.index:
        if percent_change2[i][j] < currMin:
            currMin = percent_change2[i][j]
            worstCo = i
            worstDate = j

print (fullname(worstCo), worstDate, currMin)
```

    Quanta Services Inc. 2015-10-16 -0.285005695727


Quanta Services Inc. (PWR) experienced the lowest daily return of any S&P 500 stock during 2015 of -28.50% on October 16, 2015. PWR dropped from 26.21 to 18.74 over the course of the day. 

A more in-depth look at the stock's trading data reveals that its volatility in terms of shares sold jumped from about 4 million to 24 million daily transactions. The company warned that its 3rd quarter results would be weaker than expected and results might impact the fourth quarter. 

Over the course of the year, Quanta experienced poor performance, with its share price decreasing from 28.35 to 20.25. The weak results occurred due to falling oil and gas prices because of the emergence of fracking. Consequently, the energy industry's services sector in which Quanta operates also experienced significant hardship. 

### Maximum Annual Returns

We also seek to know which companies performed the best over the course of the entire year. This can help find industries that appear attractive for investment.


```python
AnnualReturn = {}
yearMax = -math.inf
for i in percent_change2:
    AnnualReturn[i] = (priceData[i][-1]-priceData[i][0])
    			/priceData[i][0]
    if AnnualReturn[i] > yearMax:
        yearMax = AnnualReturn[i]
        maxCo = i

print (yearMax, maxCo, fullname(maxCo))
```

    1.29454911968 NFLX Netflix Inc.


Over the course of 2015, Netflix achieved the highest annual returns of any stock on the S&P 500. It achieved a phenomenal return of 129.5%, over doubling its initial price. The stock started the year at \$48.80 per share and finished the year at \$114.38.

Netflix (NFLX) had an excellent 2015 because its content library increased significantly. Furthermore, the company converted its enhanced library into new customers. Netflix's growth has been accelerated by the decline of traditional cable based television entertainment. 

### Minimum Annual Returns


```python
AnnualReturn = {}
yearMin = math.inf
for i in percent_change2:
    AnnualReturn[i] = (priceData[i][-1]-priceData[i][0])
    			/priceData[i][0]
    if AnnualReturn[i] < yearMin:
        yearMin = AnnualReturn[i]
        minCo = i

print (yearMin, minCo, fullname(minCo))
```

    -0.769784749764 CHK Chesapeake Energy


In contrast, the worst returns of 2015 belonged to Chesapeake Energy who recorded a -76.9% annual return. The stock started at \$19.57 and finished the year $4.50.  

Chesapeake's poor performance was primarily due to plummeting crude oil prices. American fracking technology oversaturated the market, so prices dropped severely. Instead of cutting production in an effort to decrease expenses, Chesapeake accelerated drilling and well production which further impacted results.

### Volatility

Additionally, a stock's returns are not the only measure of its performance. A high growth stock may also experience high volatility. The measure of a stock's price changes from day to day can serve as a proxy for risk. Stocks with high volatility are more risky to invest in. Such stocks may give high returns, or they may lose much of their value. While stocks with low volatility may not return the highest yields, their performance is more stable, allowing for predictable returns on investment.


```python
def mean(x):
    return float(sum(x)) / len(x)

def std(x):
    stdev = 0.0
    for value in x:
        difference = value - mean(x)
        stdev = stdev + (difference ** 2)
    stdev = (stdev / len(x))**(1/2)
    return stdev
```

### Maximum Volatility


```python
Volatility = {}
volMax = -math.inf

for i in percent_change2:
    Volatility[i] = std(percent_change2[i])
    if Volatility[i] > volMax:
        volMax = Volatility[i]
        volMaxC = i
        
print (volMax, volMaxC, fullname(volMaxC))
```

    0.0439833807054 FCX Freeport-McMoran Cp & Gld


Freeport-McMoran Cp & Gl (FCX) stock was the most volatile of any S&P 500 firm during 2015. This is the same ticker mentioned earlier as the one with the worst single daily return. 

FCX's high volatility can be explained by many investors pumping money into the stock after a poor day. Investors attempting to follow a "buy low; sell high" strategy will often buy shares after an abnormally poor day of returns, hoping to sell once the stock price recovers to its normal levels. FCX posted had a number of such price swings after experiencing poor daily returns.


```python
Volatility = {}
volMin = math.inf

for i in percent_change2:
    Volatility[i] = std(percent_change2[i])
    if Volatility[i] < volMin:
        volMin = Volatility[i]
        volMinC = i

print (volMin, volMinC, fullname(volMinC))
```

    0.00904485366971 KO The Coca Cola Company


In contrast, Coca Cola Company (KO) was the most stable over 2015. 
KO's stability results from its status as diversified company that primarily sells carbonated beverages. By competing in markets with extremely stable demand, the company experiences little volatility in its stock price. The stock has long been seen as a safe, bluechip investment that is found frequently in retirement funds due to its predicable returns and dividends. 

## Correlations

Stocks do not move independently from each other. Often, stock prices depend not only on the firm's internal finances but also on the industry conditions as a whole. For example, when crude oil prices fall, the entire oil industry suffers. Because we can see the price movements of one stock, we could potentially identify stocks that serve as a leading indicator for other stocks. Based on this information, we can buy or sell shares depending on how we predict the stock's price will perform.

We first define a function to calculate the correlation between any two stocks in the S&P 500, then fill a correlation matrix with all of the possible correlations between stocks. An example output is given below.


```python
def corr(x,y):
    xy = sum([a*b for a,b in zip(x,y)])
    x2 = sum([i**2 for i in x])
    y2 = sum([i**2 for i in y])
    n = len(x)
    numer = (n*xy - sum(x)*sum(y))
    denom = ((n*x2 - sum(x)**2)**(1/2) * (n*y2 - sum(y)**2)**(1/2))
    correlation = numer/denom
    return correlation

correlations = {}

for i in percent_change:
    correlations[i] = {}

for i in correlations:    
    for j in percent_change:
        correlations[i][j]=[]
    
for company1 in percent_change:
    for company2 in percent_change:
        if not correlations[company1][company2]:
            x=percent_change[company1]        
            y=percent_change[company2]
            if company1 == company2:
                correlations[company1][company2] = 1
                correlations[company2][company1] = 1
            else:
                correlations[company1][company2] = corr(x,y)
                correlations[company2][company1] = corr(x,y)
                
def corr_print(company1, company2):
    print ("The correlation coefficient between {} and {} is {}."
           .format(fullname(company1), fullname(company2), 
           	correlations[company1][company2]))

corr_print("AAPL", "MMM")   
```

    The correlation coefficient between 
    Apple Inc. and 3M Company is 0.5157280000348696.


### Most and Least Correlated Companies

For a certain company on the S&P 500 we want to find its most and least correlated stocks. When a stock moves either up or down in price, we can then assume that the most correlated stocks will follow similarly. Conversely, the least correlated stocks will not have any predictive power on the stock's price.


```python
ticker_symbols = list(priceData.columns.values)

def top_bottomcorr(ts):
    corr_tb = []
    for ss in ticker_symbols:
        if ss == ts:
            continue
        corr_co = correlations[ts][ss]
        corr_tb.append((corr_co, ss))
    corr_tb.sort()    
    print ("Most Correlated:", fullname(corr_tb[-1][1]), 
    		"(", corr_tb[-1][0],")")
    print ("Least Correlated:", fullname(corr_tb[0][1]), 
    		"(", corr_tb[0][0],")")
```

### Amazon


```python
top_bottomcorr("AMZN")
```

    Most Correlated: Alphabet Inc Class A ( 0.58555313236 )
    Least Correlated: Stericycle Inc ( 0.0564506179566 )


Amazon Stock (AMZN) is most correlated with Alphabet stock with at correlation of 0.586. Because Alphabet (formerly Google) and Amazon both compete in the technology sector, their stock movements will follow similar patterns. In contrast, Stericycle, a medical waste disposal company, has very little to do with Amazon as a firm. Waste disposal and technology share no industry trends in common, so the only price movements have to do with the economy as a whole.

### Microsoft


```python
top_bottomcorr("MSFT")
```

    Most Correlated: Marsh & McLennan ( 0.604548883151 )
    Least Correlated: Stericycle Inc ( 0.0288867601988 )


Microsoft stock is most correlated with Marsh & McLennan, with a correlation of .605. While most people would view Microsoft as a software giant, producing Windows and Office, the company also competes heavily in the IT services sector. Microsoft competes against Marsh & McLennan in the technology consulting industry, so major news regarding IT would affect both companies. Interestingly enough, the company least correlated with Microsoft is also Stericycle, the medical waste disposal firm.

### Apple


```python
top_bottomcorr("AAPL")
```

    Most Correlated: Illinois Tool Works ( 0.601265434285 )
    Least Correlated: Range Resources Corp. ( 0.112710875584 )


In a very interesting result, the company most correlated with Apple in 2015 is Illinois Tool Works, with a correlation of 0.601. The result is likely coincidental because the two companies lie in two separate industries with little to do with each other. The association is probably an outlier and the correlation should decrease in the future. The company least correlated with Apple is Range Resources, a natural gas exploration firm. This would make sense because the energy industry has little relationship with technology. Investors looking to diversify away from Apple should look into the oil and gas sector.

### Facebook


```python
top_bottomcorr("FB")
```

    Most Correlated: Fiserv Inc ( 0.61966671131 )
    Least Correlated: Newmont Mining Corp. ( -0.00283227001625 )


Facebook (FB) is found to be most strongly correlated with Fiserv Inc (FISV) in terms of stock return, with the correlation coefficient of 0.620. Both firms operate in technology although Fiserv focuses primarily on the financial sector. Fiserv has built a banking platform on Facebook for companies to build applications on Facebook. The link might explain their correlated stock returns.

As we have previously encountered, the stock least correlated with Facebook is an energy firm. Mining giant Newmont Mining Corp has essentially no correlation with Facebook. As discussed previously, technology and energy appear to be wholly uncorrelated fields.

### Google


```python
top_bottomcorr("GOOGL")
```

    Most Correlated: Alphabet Inc Class C ( 0.989365040395 )
    Least Correlated: Transocean ( 0.00952277504792 )


Google (GOOGL) is most closely correlated with Alphabet Inc Class C (GOOG) with correlation coefficient of 0.989. This is expected because Google split its stock into two parts: Alphabet Inc Class A and Alphabet Inc Class C. Since GOOGL and GOOG essentially belong to the same company, their stock returns behave almost exactly the same.

The least correlated stock with Google is Transocean, an offshore drilling contractor for crude oil firms. Again we see that energy and technology do not correlate with stock returns.

## Clustering Algorithm

Once we know which stocks correlate with each other, we want to group the stocks into sets of highly correlated firms. Each resulting cluster will move in a similar manner throughout the year. We can do this to identify industries and firms that are highly interrelated.

First, we transform our correlation matrix into a set of tuples. We first want to sort our list of correlations from highest to lowest so we can group the companies together with the highest correlations. We implement the sorting using margeSort, an algorithm with O(n log(n)) time complexity. 

Because the correlation matrix is symmetric, we only need to sort one half of the correlations. Additionally, the diagonals of the correlation matrix are 1, so we can ignore them. 


```python
correlations = percent_change.corr()

correlations = correlations.where(np.triu(np.ones(correlations.
		shape)).astype(np.bool))
correlations = correlations.stack().reset_index()
correlations.columns = ['Company1', 'Company2', 'Correlation']

correlation_tuples = [tuple(x) for x in correlations.values]
```


```python
def mergeSort(array):
    if len(array) > 1:
        mid = len(array) //2
        left = array[:mid]
        right = array [mid:]
        mergeSort(left)
        mergeSort(right)

        i = 0
        j = 0
        k = 0
        while i < len(left) and j < len(right):
            if left[i][2] > right[j][2]:
                array[k] = left[i]
                i = i + 1
            else:
                array[k] = right[j]
                j = j + 1
            k = k+1
        while i < len(left):
            array[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            array[k] = right[j]
            j += 1
            k += 1
    return(array)

sortedWeights = mergeSort(correlation_tuples)
```

We will represent the correlation clusters by using an agglomerating clustering algorithm. The algorithm is a modified version of Prim's or Kruskal's algorithms. The aforementioned algorithms prohibit cycles, whereas we treat a cycle as a cluster. 

We represent the graph with the nodes as each individual stock and the edges as their respective correlations.

Initially we initialize each node to have one edge, where the edge points back to the node itself. We label such nodes as bottom nodes. Each node where nothing besides itself points to it is known as a starting node. 

The algorithm works by finding the bottom node of an interconnected set. By starting from a node, we can trace its path to a node that will point to itself. That node we find is the bottom node of a set. From that node, we can build our set of similarly correlated stocks.


```python
# This is a simplified version of the NetworkX class DiGraph.

class Digraph():
    def __init__(self,filename = None):
        self.edges = {}
        self.numEdges = 0
    
    def addNode(self,node):       
        self.edges[node] = set()

    def add_Edge(self,src,dest,weight):
        if not self.hasNode(src): 
            self.addNode(src)
            self.edges[src] = {}
        if not self.hasNode(dest): 
            self.addNode(dest)
            self.edges[dest] = {}
        if not self.hasEdge(src, dest):
            self.numEdges += 1
            self.edges[src][dest] = weight
  
    def childrenOf(self, v):
        # Returns a node's children
        return self.edges[v].items()
        
    def hasNode(self, v):
        return v in self.edges
        
    def hasEdge(self, v, w):
        return w in self.edges[v]

    def listEdges(self):
        ll = []
        for src,values in self.edges.items():
            for dest,weight in values.items():
                ll.append([src,dest,weight])
        return ll
    
    def __str__(self):
        result = ''
        for src in self.edges:
            for dest,weight in self.edges[src].items():
                result = result + src + '->'\
                         + dest + ', length ' + str(weight) + '\n'
        return result[:-1]

class Graph(Digraph):
    def addEdge(self, src, dest, weight):
        Digraph.addEdge(self, src, dest, weight)
        Digraph.addEdge(self, dest, src, weight)
```


```python
def init_graph(sortedWeights):
    graph = Graph()
    for x in sortedWeights:
         graph.add_Edge(x[0],x[1],weight = x[2])
    return graph

def init_nodePointers(graph):
    nodePointers = {src:src for src in graph.edges}
    return nodePointers

def init_nodeStarting(graph):
    nodeStarting = {src:True for src in graph.edges}
    return nodeStarting

def init_nodeBottom(graph):
    nodeBottom = {src:True for src in graph.edges}
    return nodeBottom

def findbottom(node, nodePointers):
    source = node
    destination = nodePointers[source]
    while destination != source:
        source = destination
        destination = nodePointers[source]
    return destination

def mergeSets(sortedWeights, k):
    sortedWeights = [value for value in sortedWeights 
    				if value[0] != value[1]]
    graph = init_graph(sortedWeights)
    nodePointers = init_nodePointers(graph)
    nodeStarting = init_nodeStarting(graph)
    nodeBottom = init_nodeBottom(graph)
    counter = 0
    for key in sortedWeights:
        if counter < k:
            bottom1 = findbottom(key[0], nodePointers)
            bottom2 = findbottom(key[1], nodePointers)
            if bottom1 != bottom2:
                nodePointers[bottom2] = bottom1
                nodeBottom[bottom2] = False
                nodeStarting[bottom1] = False
                
            counter += 1
    return (nodePointers, nodeStarting, nodeBottom)

def recoverSets(nodePointers, nodeStarting, nodeBottom):
    dict = {}
    for b_key, b_value in nodeBottom.items():
        if b_value:
            dict.setdefault(b_key, set())
            for s_key, s_value in nodeStarting.items():
                if s_value and findbottom(s_key, nodePointers)
            == b_key:
                    bottom = findbottom(s_key, nodePointers)
                    current_node = s_key
                    while current_node != bottom:
                        dict[b_key].add(current_node)
                        current_node = nodePointers[current_node]
            dict[b_key].add(b_key)
    return list(dict.values())
```

We can now run our algorithm k times. K represents the number of edges we will add. These edges will transition the node pointers from one node to another. We return the clusters the algorithm finds. 

When we pass the algorithm k = 10,000, that is the number of edges we allow it to modify, the stock market is clustered into 1 giant clump. This makes sense, since there are at most ((n)*(n-1)) / 2 < 100000 edges, so the entire graph is combined into one cluster.


```python
nodePointers, nodeStarting, nodeBottom 
				= mergeSets(sortedWeights, 100000)
print(recoverSets(nodePointers, nodeStarting, nodeBottom))
```

In contrast, if we set k = 0, then each stock would be clustered in its own set, which is not useful information either. Therefore, we have to find a suitable value for k such that the stocks will be in small, useful clusters. We settle on k = 2000, which gives us a balance between clusters with only one or two stocks and a single giant cluster of the entire S&P 500 index. 

What we notice is that many firms reside on their own, barely correlated with any other firm in the index. This makes sense because a number of industries only have one company listed. Additionally, there exists a large cluster of firms. The massive group represents the American economy as a whole. As these stocks move, so too does the United States' economic health. 


```python
nodePointers, nodeStarting, nodeBottom 
			= mergeSets(sortedWeights, 2000)
cluster_2000 = recoverSets(nodePointers, nodeStarting, nodeBottom)
print(cluster_2000)
print("For k = 2000, " + str(len(cluster_2000)) + 
			" clusters are generated." + '\n')
```

    For k = 2000, 218 clusters are generated.
    

An example of clusters at k = 2000 includes {'DAL', 'AAL', 'LUV', 'UAL', 'ALK'}. This cluster contains all airlines, which would make sense that their returns are significantly correlated. Fuel spikes, travel delays, and passenger volumes affect the entire aviation industry, so all airlines within it would have similar returns. 

Another cluster at is {'UNH', 'CI', 'AET', 'CNC', 'ANTM'}. This cluster contains all health insurance or pharmaceutical companies. Because of the beginning of the Affordable Care Act (Obamacare) the health insurance and pharmaceutical companies suddenly had many more patients to insure. Additionally, the way medicine was being paid for shifted due to the law coming into effect. Therefore, the health industry is significantly correlated due to the Affordable Care Act.

A third cluster includes {'MAR', 'HOT', 'WYN'}, three large chains of hotels. Similarly to airlines, the industry operates heavily based on seasonal volume, local events, and travel habits on national holidays such as July 4th, Labor Day, and Memorial Day. Christmas and Easter travel habits also strongly affect the hotels industry. Because the three stocks are direct competitors, they face similar returns in relation to the market.

A further cluster within the groupings includes {'VLO', 'PSX', 'TSO', 'MPC'}, all oil companies competing heavily in the fracking industry. Due to oil prices plummeting due to decreased demand and a boom in supply from shale oil and fracking, these companies faced a struggle for revenues. Their stocks decreased significantly throughout the year, noted by sharp declines as OPEC held meetings to keep production steady.

Another cluster within the S&P 500 returns is a series of railroads, {'KSU', 'NSC', 'UNP', 'CSX'}. Because the industry operates in an oligopoly due to the difficulty of installing rail track, the firms are all closely linked in strong rivalry. Market pressures such as the West Coast port strikes and decreased demand for goods shipped inland from China affected their returns equally, leading to highly correlated stocks.

The large entertainment studios in the S&P 500 are also strongly correlated with each other. {'FOX', 'FOXA', 'DIS', 'CMCSA', 'TWX'} experienced similarly correlated returns primarily because the entire industry operates on a seasonal basis, with new shows appearing in the fall and prices moving accordingly. In addition the industry as a whole faces pressure from industry disruptor Netflix, a company that is causing traditional forms of entertainment to decline.

The final cluster highlighted is one comprising of three tobacco firms, {'MO', 'PM', 'RAI'}. Their returns would be strongly correlated by the release of potentially negative studies showing the risks of smoking on a person's health.

In general, the clustering algorithm mirrors the S&P 500 well. Companies in the same industry, usually direct competitors, have highly correlated returns. Companies that do not have a counterpart on the index are in their own cluster, generating returns relatively independently. This is why portfolio diversification works, as individual stocks in separate industries are less correlated. Finally, there exists a large cluster of stocks, signifying general industry. These stocks represent the a snapshot of the American economy as a whole, moving along with the market.

## Prim's and Kruskal's Algorithm

The algorithm that we have executed above is Kruskal's Algorithm. We start with a forest of one node trees, and we join the trees greedily, starting with the edges with the least cost or highest benefit, in this example highest correlation, for up to k times. 

Kruskal's algorithm is a greedy algorithm that generates a Minimum Spanning Tree (MST) given a set of nodes and edges. Our algorithm works very similarly to Kruskal's algorithm, except that our algorithm terminates after k iterations, and not when a spanning tree is formed.

Our algorithm also tries to generate "trees" where each tree is a cluster. Our end result is a forest that consists of a number of trees. In our algorithm, we have prevented any cycles from forming. If the two nodes connected to an edge are in the same cluster, then they will have the same "bottom" node. Our algorithm will then make no changes. Similarly, if an edge connects two nodes within a tree, it will not add the edge to the tree, or it will form a cycle.

## Returns Plots

Once we have identified our clusters, we want to plot their returns to verify that they move similarly. Based on the blots below, we can see that the returns of the stocks within each cluster are very similar on a daily level. Such charts indicate that the algorithm has created sensible clusters because of the presence of similar peaks and dips.


```python
percent_change[['DAL', 'AAL', 'LUV', 'UAL', 'ALK']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_69_1.png){:class="img-responsive"}


```python
percent_change[['UNH', 'CI', 'AET', 'CNC', 'ANTM']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_70_1.png){:class="img-responsive"}


```python
percent_change[['MAR', 'HOT', 'WYN']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_71_1.png){:class="img-responsive"}


```python
percent_change[['VLO', 'PSX', 'TSO', 'MPC']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_72_1.png){:class="img-responsive"}


```python
percent_change[['KSU', 'NSC', 'UNP', 'CSX']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_73_1.png){:class="img-responsive"}


```python
percent_change[['FOX', 'FOXA', 'DIS', 'CMCSA', 'TWX']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_74_1.png){:class="img-responsive"}


```python
percent_change[['MO', 'PM', 'RAI']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_75_1.png){:class="img-responsive"}

## Normalized Prices

Similarly, for normalized price movement (each stock starts at price = 1 at the start of the year), the stocks within each cluster seems to perform closely. Within the airline cluster, ALK posts significantly better than the other firms in the cluster. Its general shape, however, mimics the other stocks in the group with declines and upticks at similar periods. While the returns for some stocks within each group might outperform (or underperform) others, the timing of movements in either direction occurs on the same day. These significant movements in price occur because of an event that affected the entire industry, so all stocks within the cluster move similarly.


```python
pricesScaled = priceData.divide(priceData.ix[0]) 

pricesScaled[['DAL', 'AAL', 'LUV', 'UAL', 'ALK']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_77_1.png){:class="img-responsive"}


```python
pricesScaled[['UNH', 'CI', 'AET', 'CNC', 'ANTM']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_78_1.png){:class="img-responsive"}


```python
pricesScaled[['MAR', 'HOT', 'WYN']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_79_1.png){:class="img-responsive"}


```python
pricesScaled[['VLO', 'PSX', 'TSO', 'MPC']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_80_1.png){:class="img-responsive"}


```python
pricesScaled[['KSU', 'NSC', 'UNP', 'CSX']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_81_1.png){:class="img-responsive"}


```python
pricesScaled[['FOX', 'FOXA', 'DIS', 'CMCSA', 'TWX']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_82_1.png){:class="img-responsive"}


```python
pricesScaled[['MO', 'PM', 'RAI']].plot()
```

![image1](assets/posts/2017-01-23-Stock-Clustering_files/2017-01-23-Stock-Clustering_83_1.png){:class="img-responsive"}

*If you've made it down this far, I'm really impressed!!! Hope you enjoyed my first post!* :)
