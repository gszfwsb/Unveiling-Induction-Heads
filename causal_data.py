from typing import Any, Dict
import torch
import math


class DataMethod:
    def __init__(self, config: Dict = None):
        """
        Initialize the DataMethod.

        Args:
            config (Dict, optional): Configuration parameters for the data method. Defaults to None.
        """
        if config is None:
            config = {}
        self.config = config

    def generate_data(self, **kwargs) -> Any:
        """
        Stub for data generation method.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Generated data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def transform_data(self, data: Any, **kwargs) -> Any:
        """
        Stub for data transformation method.

        Args:
            data (Any): The data to transform.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Transformed data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")



class GraphCausalModel(DataMethod):
    # write me an initializer
    def __init__(self, dict: Dict = None):
        """
        Initialize the GraphCausal method.

        Args:
            dict (Dict): The parameters for the data generation.
            dict['A_type']: str, type of adjacent matrix, 'Markov chain' or 'Two grams'...
            dict['T']: int, sequence length
            dict['dx']: int, dimension of x
            dict['dy']: = 0
        """
        super().__init__(dict)
        self.T = dict["T"] + 1 # add one to the sequence length 'cause the last token is the target
        self.dx = dict['dim']
        self.d = self.dx 
        self.number_of_samples = dict.get('number_of_samples', 1)
        
        # initialize A matrix with lower triangular matrix with one on 2nd diagonal
        if dict['A_type'] == 'Markov chain':
            self.A = torch.diag(torch.ones(self.T-1), diagonal=-1)
            self.A[0:1, :] = 0
        if dict['A_type'] == 'Two grams':
            self.A = (torch.diag(torch.ones(self.T-1), diagonal=-1)  
            + torch.diag(torch.ones(self.T-2), diagonal=-2))
            self.A[0:2, :] = 0

        # extract the position where there is no parent
        self.no_parent = torch.where(self.A.sum(1) == 0)[0]
        self.with_parent = torch.where(self.A.sum(1) != 0)[0]
        
       

    def __generatedata__(self, **kwargs) -> Any:
        """
        Generate causal data.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The generated data.
        """
        # Assume parent number to be equal value or 0. 
        pn = int(torch.max(torch.sum(self.A, dim=1)))
       


        # generate x as follows: for each sequence self.T, sample from uniform self.d
        # if row sum of A is 0, then sample from uniform self.d
        # if row sum of A is pn, then sample from P given the previous pn tokens
        x = torch.zeros(self.number_of_samples, self.T, self.dx)
        for i in range(self.number_of_samples):
             # generate transition matrix self.d by self.d^p
            P = torch.rand(self.d, self.d**pn)
            P = P / P.sum(1, keepdim=True)

            for j in range(self.T):
                # if no parent, sample from uniform self.d
                if self.A[j,:].sum() == 0:
                    x[i, j, torch.randint(self.dx, (1,))] = 1
                if self.A[j,:].sum() == pn:
                    # find the parents of j
                    parents_idx = torch.nonzero(self.A[j, :]).squeeze(1)
                    # for each idx, find the corresponding token in x
                    parents = x[i, parents_idx, :]
                    # find the corresponding column in P
                    col = 0
                    for k in range(pn):
                        # find index of non-zero element in parents
                        parent_state = torch.nonzero(parents[k]).squeeze()
                        # find the corresponding column in P
                        # print("col" , parent_state * self.d**(pn-k-1))
                        col += parent_state * self.d**(pn-k-1)


                    # sample from P given the previous pn tokens
                    x[i, j, torch.multinomial(P[:, col], 1)] = 1
        
        return x.squeeze(0)
    
                
    def __transform__(self, x: Any, **kwargs) -> Any:
        """
        Transform the data for training, validation, and testing.

        Args:
            x (Any): The data.

        Returns:
            Any: The transformed data.
        """
        y = x[..., -1, :].clone() # remember to clone the tensor!!!
        x = x[..., :-1, :]
        return x, y

        
# create test function for CausalModel.generate_data
def test_generatedata():
    
    # Set the required attributes for testing
    c_dict = {
        'A_type': 'Markov chain',
        'T': 20,
        'dim': 10,
        'number_of_samples': 2
    }
    data_method = GraphCausalModel(c_dict)
    # Call the __generatedata__ method
    generated_data = data_method.__generatedata__()
    # Print the generated data
    


# if __name__ == "__main__":
#     test_generatedata()