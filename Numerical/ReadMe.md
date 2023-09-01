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

## Gradient descent summary
So far, you have developed a linear model that predicts $f_{w_j,b}(x^{(i)})$:
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{3}$$
In linear regression, you utilize input training data to fit the parameters $w$,$b$ by minimizing a measure of the error 
between our predictions $f_{w,b}(x^{(i)})$ and the actual data $y^{(i)}$. The measure is called the $cost$, $J(w,b)$. 
In training, you measure the cost over all of our training samples $x^{(i)},y^{(i)}$
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{4}$$ 

*Gradient descent* was described as:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{5}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously.  
The gradient is defined as:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}\\ 
\tag{6}
\frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{7}\\
\end{align}
$$

Here *simultaniously* means that you calculate the partial derivatives for all the parameters before updating 
any of the parameters.

## Implement Gradient Descent
You will implement gradient descent algorithm for one feature. You will need three functions. 
- `compute_gradient` implementing equation (6) and (7) above
- `compute_cost` implementing equation (4) above (code from previous lab)
- `gradient_descent`, utilizing compute_gradient and compute_cost

Conventions:
- The naming of python variables containing partial derivatives follows this pattern, 
$\frac{\partial J(w,b)}{\partial b}$  will be `dj_db`.
- w.r.t is With Respect To, as in partial derivative of $J(wb)$ With Respect To $b$.

### compute_gradient
`compute_gradient`  implements (6) and (7) above and returns 
$\frac{\partial J(w,b)}{\partial w}$,$\frac{\partial J(w,b)}{\partial b}$. 
The embedded comments describe the operations.

###  Gradient Descent
Now that gradients can be computed,  gradient descent, described in equation (5) above can be implemented below 
in `gradient_descent`. The details of the implementation are described in the comments. 
Below, you will utilize this function to find optimal values of $w$ and $b$ on the training data. 



