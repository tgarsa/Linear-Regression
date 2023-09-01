# Numerical Approach

The idea here is to use a purely mathematical approach, computing the quadratic error of each prediction 
and trying to reduce this error. 

To reduce the error, we will calculate the derivatives of the prediction function. 
For this reason, we need to define the model in advance; in this case, it is a linear regression, 
and we derive the function in terms of the w_i and b parameters. 

In addition, we used an $\alpha$ parameter to define the step size of each interaction.

## Computing Cost
The term 'cost' in this assignment might be a little confusing since the data. 
Here, cost is a measure how well our model is predicting the target feature.

The equation for cost with multiples variable is:
  $$J(w_j,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w_j,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
 
where 
  $$f_{w_j,b}(X^{(i)}) = \sum\limits_{j = 0}^{n-1} w_jx_j^{(i)} + b \tag{2} $$
  
- $f_{w_j,b}(X^{(i)})$ is our prediction for example $i$ using parameters $w_j,b$.  
- $(f_{w_j,b}(X^{(i)}) -y^{(i)})^2$ is the squared difference between the target value and the prediction.   
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w_j,b)$.  


