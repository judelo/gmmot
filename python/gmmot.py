import numpy as np
import ot
import scipy.stats as sps
import scipy.linalg as spl
from scipy.optimize import linprog
import matplotlib.pyplot as plt

#################
### author : Julie Delon
#################

###############################
#### display GMM
###############################


def display_gmm(gmm,n=50,ax=0,bx=1,ay=0,by=1,cmap='viridis',axis=None):
    
    if axis is None:
        axis = plt.gca()
        
    [K,pi,mu,S] = gmm
    
    x = np.linspace(ax, bx,num=n)
    y = np.linspace(ay, by,num=n)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = densite_theorique2d(mu,S,pi,XX)
    Z = Z.reshape(X.shape)
    plt.axis('equal')
    return axis.contour(X, Y, Z,8,cmap=cmap)    





###############################
#### compute GMM densities
###############################

def densite_theorique(mu,var,alpha,x):
    # compute the 1D GMM density with parameters (mu,var) and weights alpha  at x 
    K=mu.shape[0]
    y=0
    #y=np.zeros(len(x))
    for j in range(K):
        y+=alpha[j]*sps.norm.pdf(x,loc=mu[j,:],scale=np.sqrt(var[j,:,:]))
    return y.reshape(x.shape)

def densite_theorique2d(mu,Sigma,alpha,x):
    # compute the 2D GMM density with parameters (mu, Sigma) and weights alpha at x
    K = mu.shape[0]
    alpha = alpha.reshape(1,K)
    y=0
    for j in range(K):
        y+=alpha[0,j]*sps.multivariate_normal.pdf(x,mean=mu[j,:],cov=Sigma[j,:,:])
    return y

###############################
### Optimal Transport between Gaussians (quadratic Wasserstein)
###############################

def GaussianW2(m0,m1,Sigma0,Sigma1):
    # compute the quadratic Wasserstein distance between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1
    Sigma00  = spl.sqrtm(Sigma0)
    Sigma010 = spl.sqrtm(Sigma00@Sigma1@Sigma00)
    d        = np.linalg.norm(m0-m1)**2+np.trace(Sigma0+Sigma1-2*Sigma010)
    return d

def GaussianMap(m0,m1,Sigma0,Sigma1,x):
    # Compute the OT map (evaluated at x) between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1 
    # m0 and m1 must be 2D arrays of size 1xd
    # Sigma0 and Sigma1 must be 2D arrays of size dxd
    # x can be a matrix of size n x d,
    # each column of x is a vector to which the function is applied
    d = Sigma0.shape[0]
    m0 = m0.reshape(1,d)
    m1 = m1.reshape(1,d)
    Sigma0 = Sigma0.reshape(d,d)
    Sigma1 = Sigma1.reshape(d,d)
    Sigma  = np.linalg.inv(Sigma0)@spl.sqrtm(Sigma0@Sigma1)
    Tx        = m1+(x-m0)@Sigma
    return Tx

def GaussianBarycenterW2(mu,Sigma,alpha,N):
    # Compute the W2 barycenter between several Gaussians
    # mu has size Kxd, with K the number of Gaussians and d the space dimension
    # Sigma has size Kxdxd
    K        = mu.shape[0]  # number of Gaussians
    d        = mu.shape[1]  # size of the space
    Sigman   = np.eye(d,d)
    mun      = np.zeros((1,d))
    cost = 0
    
    for n in range(N):
        Sigmandemi       = spl.sqrtm(Sigman)
        T = np.zeros((d,d))
        for j in range(K):
            T+= alpha[j]*spl.sqrtm(Sigmandemi@Sigma[j,:,:]@Sigmandemi)
        Sigman  = T
    
    for j in range(K):
        mun+= alpha[j]*mu[j,:]
    
    for j in range(K):
        cost+= alpha[j]*GaussianW2(mu[j,:],mun,Sigma[j,:,:],Sigman)

    return mun,Sigman,cost       # return the Gaussian Barycenter (mun,Sigman) and the total cost


###############################
###### GW2 between GMM
###############################


def GW2(pi0,pi1,mu0,mu1,S0,S1):
    # return the GW2 discrete map and the GW2 distance between two GMM
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    d  = mu0.shape[1]
    S0 = S0.reshape(K0,d,d)
    S1 = S1.reshape(K1,d,d)
    M  = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library
    wstar     = ot.emd(pi0,pi1,M)         # discrete transport plan
    distGW2   = np.sum(wstar*M)
    return wstar,distGW2

def GW2cost(mu0,mu1,S0,S1):       # return the distance matrix M of size K0 x K1
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    M = np.zeros((K0,K1))
    # we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    return M

def GW2_map(pi0,pi1,mu0,mu1,S0,S1,wstar,x):      
    # return the GW2 maps between two GMM on the 1D grid x  
    n,K0,K1    = x.shape[0],mu0.shape[0],mu1.shape[0]
    T          = np.zeros((K0,K1,n))     # each Tkl = T[k,l,:] is of dimension n and correspond to the W2-map between component k of mu0 and component l of mu1
    tmpmean    = np.zeros(n)
    weightmean = np.zeros(n)
    Tmean      = np.zeros((n,n))     # averaged map on a grid 
    Tmap       = np.zeros((n,n))     # multivalued map on a grid
    
    for k in range(K0):
        for l in range(K1):
            if wstar[k,l]!=0:
                T[k,l,:] = GaussianMap(mu0[k,:],mu1[l,:],S0[k,],S1[l],x).reshape(n,)
                for i in range(n):
                    Ti             = int(max(min(T[k,l,i],1),0)*99)
                    Tmap[i,Ti]    += wstar[k,l]*sps.norm.pdf(x[i],loc=mu0[k],scale=np.sqrt(S0[k]))
                    tmpmean[i]    += wstar[k,l]*sps.norm.pdf(x[i],loc=mu0[k],scale=np.sqrt(S0[k]))/densite_theorique(mu0,S0,pi0,x[i])*T[k,l,i]
                    weightmean[i] += wstar[k,l]*sps.norm.pdf(x[i],loc=mu0[k],scale=np.sqrt(S0[k]))

    tmpmean = np.uint(np.maximum(np.minimum(tmpmean,1),0)*99)
    for i in range(n):
        Tmean[i,tmpmean[i]] = weightmean[i]
    
    return Tmap,Tmean


#####################################################
##### Multimarginal problem
#####################################################


def create_cost_matrix_from_gmm(gmm,alpha,N=10):
    """
    create the cost matrix for the multimarginal problem between all GMM
    create the barycenters (mun,Sn) betweenn all Gaussian components 
    """
    
    nMarginal       = len(alpha)               # number of marginals
    d               = gmm[0][2].shape[1]       # space dimension
    tup = ();
    for k in range(nMarginal):
        K  = gmm[k][0]
        tup+=(K,)        
    C               = np.zeros(tup)
    mun             = np.zeros(tup+(d,))
    Sn             = np.zeros(tup+(d,d))
        
    it = np.nditer(C,["multi_index"])
    while not it.finished:
        tup = it.multi_index        
        mu = np.zeros((nMarginal,d))
        Sigma = np.zeros((nMarginal,d,d))
        
        for k in range(nMarginal):
            mu[k,:]      = gmm[k][2][tup[k]]
            Sigma[k,:,:] = gmm[k][3][tup[k]]
            
        mu    = np.array(mu)    
        Sigma = np.array(Sigma)    
        [mun[tup],Sn[tup],cost] = GaussianBarycenterW2(mu,Sigma,alpha,N)
        
        C[tup] = cost
        
        it.iternext()
    
    return C,mun,Sn  

def solveMMOT(pi, costMatrix, epsilon = 1e-10):
    """ Author : Alexandre Saint-Dizier
    
    Solveur of the MultiMargnal OT problem, using linprog

    Input :
     - pi : list(array) -> weights of the different distributions
     - C : array(d1,...dp) -> cost matrix
     - epsilon : smallest value considered to be positive

    Output :
     - gamma : list of combinaison with positive weight
     - gammaWeights : corresponding weights
    """

    nMarginal = len(pi);
    nPoints = costMatrix.shape;

    nConstraints = 0; nParameters = 1;
    for ni in nPoints:
        nConstraints += ni; nParameters *= ni

    index = 0;
    A = np.zeros((nConstraints, nParameters)); b = np.zeros(nConstraints)
    for i in range(nMarginal):
        ni = nPoints[i];
        b[index:index+ni] = pi[i];
        for k in range(ni):
            Ap = np.zeros(costMatrix.shape);
            tup = ();
            for j in range(nMarginal):
                if j==i:
                    tup+= (k,)
                else:
                    tup+=(slice(0,nPoints[j]),)
            Ap[tup] = 1;
            A[index+k,:]=Ap.flatten();
        index += ni
    A = A.tolist(); b = b.tolist();
    C = costMatrix.flatten().tolist()

    res = linprog(C, A_eq=A, b_eq =b) #Solve inf <C,X> with constraints AX=b
    gammaWeights = res.x;
    gammaWeights = gammaWeights.reshape(costMatrix.shape)
   
    return gammaWeights







####################################################
#### for color transfer or color barycenters   #####
#### guided_filter is used for post-processing #####
####################################################


def average_filter(u,r):
    # uniform filter with a square (2*r+1)x(2*r+1) window
    # u is a 2d image
    # r is the radius for the filter
    
    (nrow, ncol)                                      = u.shape
    big_uint                                          = np.zeros((nrow+2*r+1,ncol+2*r+1))
    big_uint[r+1:nrow+r+1,r+1:ncol+r+1]               = u
    big_uint                                          = np.cumsum(np.cumsum(big_uint,0),1)       # integral image
    
    out = big_uint[2*r+1:nrow+2*r+1,2*r+1:ncol+2*r+1] + big_uint[0:nrow,0:ncol] - big_uint[0:nrow,2*r+1:ncol+2*r+1] - big_uint[2*r+1:nrow+2*r+1,0:ncol]
    out = out/(2*r+1)**2
    
    return out

def guided_filter(u,guide,r,eps):
    C           = average_filter(np.ones(u.shape), r)   # to avoid image edges pb
    mean_u      = average_filter(u, r)/C
    mean_guide  = average_filter(guide, r)/C
    corr_guide  = average_filter(guide*guide, r)/C
    corr_uguide = average_filter(u*guide, r)/C
    var_guide   = corr_guide - mean_guide * mean_guide
    cov_uguide  = corr_uguide - mean_u * mean_guide
    
    alph = cov_uguide / (var_guide + eps)
    beta = mean_u - alph * mean_guide
    
    mean_alph = average_filter(alph, r)/C
    mean_beta = average_filter(beta, r)/C
    
    q = mean_alph * guide + mean_beta
    return q


