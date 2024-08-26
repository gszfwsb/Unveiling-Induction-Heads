import torch
import unittest

class TestEinsumOperation(unittest.TestCase):
    
    def setUp(self):
        # Set up test data
        self.norm_query = torch.randn(2, 3, 4)  # Shape [b, h, q]
        self.norm_key = torch.randn(2, 3, 5)    # Shape [b, h, k]
    
    def test_einsum_operation(self):
        # Perform the einsum operation
        result = torch.einsum('bhq,bhk->bhqk', self.norm_query, self.norm_key)
        
        # Assert the result shape
        expected_shape = (2, 3, 4, 5)
        self.assertEqual(result.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()