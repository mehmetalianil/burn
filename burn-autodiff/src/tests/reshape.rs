#[burn_tensor_testgen::testgen(ad_reshape)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_reshape() {
        let tensor_1 = TestAutodiffTensor::from([[1.0, 7.0], [2.0, 3.0]]).require_grad();
        let tensor_2 = TestAutodiffTensor::from([4.0, 7.0, 2.0, 3.0]).require_grad();

        let tensor_3 = tensor_2.clone().reshape([2, 2]);
        let tensor_4 = tensor_1.clone().matmul(tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([3.0, 3.0, 10.0, 10.0]));
    }
}
