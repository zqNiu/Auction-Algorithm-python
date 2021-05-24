import numpy as np
from scipy.optimize import linear_sum_assignment
import time
def auction_asy(cost_matrix,minimize=False):
    """Solve the linear sum assignment problem according to Bertsekas's paper

    arameters
    ----------
    cost_matrix : array
        The cost matrix of the assignment.
    maximize : bool (default: False)
        Calculates a maximum weight matching if true.
    Returns
    -------
    assign : array
        corresponding indices giving the optimal assignment
        index: person index     value: object index. 
    obj: float
        objective function 

    References
    ----------
    1. [Bertsekas's Auction Algorithm](http://dspace.mit.edu/bitstream/handle/1721.1/3233/P-2064-24690022.pdf?sequence=1). 
    The algorithm solves the problem of optimally assigning N objects to N people given the preferences
    specified in a given cost matrix.
    2. https://github.com/scipy/scipy/blob/v1.6.3/scipy/optimize/_lsap.py#L16-L105
    3. https://github.com/EvanOman/AuctionAlgorithmCPP
    """
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.ndim != 2:
        raise ValueError("expected a matrix (2-D array), got a %r array"
                         % (cost_matrix.shape,))

    if not (np.issubdtype(cost_matrix.dtype, np.number) or
            cost_matrix.dtype == np.dtype(np.bool_)):
        raise ValueError("expected a matrix containing numerical entries, got %s"
                         % (cost_matrix.dtype,))

    if minimize:
        cost_matrix = -cost_matrix

    cost_matrix = cost_matrix.astype(np.double)

    # auction algorithm implement 
    N=99999
    num_person=num_object=cost_matrix.shape[0]
    assign=np.asarray([-N]*(num_person))
    epsilon=10
    price=np.asarray([1]*(num_object))

    while(epsilon >= 0.8):
        assign=np.asarray([-N]*(num_person))
        time_start=time.time()
        while ((assign<0).any()):
            #bidding phase
            #Compute the bids of each unassigned individual person and store them in temp array
            bid_value=np.zeros((num_person,num_object))  # row:person col:object
            best_margin=-N   
            best_margin_j_index=-N
            second_margin=-N
            second_margin_j_index=-N
            for i in range(num_person):
                if assign[i]<0:
                    # unassigned
                    # Need calculate the best(max) and second best(max) value of each object to this person
                    for j in range(num_object):
                        margin=cost_matrix[i][j]-price[j]
                        if margin>best_margin:
                            best_margin=margin
                            best_margin_j_index=j
                        elif margin>second_margin:
                            second_margin=margin
                            second_margin_j_index=j

                    bid_value[i][best_margin_j_index]=cost_matrix[i][best_margin_j_index]-\
                                                      second_margin+epsilon
                    # also =price[best_margin_j_index]+best_margin-second_margin+epsilon
            #assignment phase
            #Each object which has received a bid determines the highest bidder and 
            #updates its price accordingly
            bid_value_T=np.transpose(bid_value)  # row:object col:person
            for j in range(num_object):
                bid_for_j=bid_value_T[j]
                if((bid_for_j>0).any()):
                    max_bid=np.max(bid_for_j)
                    max_bid_person=np.where(bid_for_j==np.max(bid_for_j))[0][0]
                    if np.where(assign==j)[0].shape[0]>0:
                        # j has been assigned, corresponding i need be set to unassigned 
                        i_index=np.where(assign==j)[0][0]
                        assign[i_index]=-N
                    assign[max_bid_person]=j
                    price[j]=max_bid
        epsilon=epsilon/2
        obj=0
        for i in range(num_person):
            obj+=cost_matrix[i][assign[i]]
        print(epsilon,obj,time.time()-time_start)
        
    obj=0
    for i in range(num_person):
        obj+=cost_matrix[i][assign[i]]
    return {"assign":assign,"obj":obj}

if __name__=="__main__":
    #cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    cost = np.random.randint(low=0,high=100,size=(100,100))
    time_start1=time.time()
    result = auction_asy(cost)
    time_end1=time.time()
    print("Auction algorithm:",result["assign"],result["obj"],"running time={}".format(time_end1-time_start1))
    # compared with scipy:hungarian algorithm
    time_start2=time.time()
    row_ind, col_ind = linear_sum_assignment(cost,True)
    time_end2=time.time()
    print("hungarian algorithm:",col_ind,cost[row_ind, col_ind].sum(),"running time={}".format(time_end2-time_start2))