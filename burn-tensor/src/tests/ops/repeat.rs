#[burn_tensor_testgen::testgen(repeat)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn should_support_repeat_ops() {
        let tensor = Tensor::<TestBackend, 2>::from([[0.0, 1.0, 2.0]]);

        let data_actual = tensor.repeat(0, 4).into_data();

        let data_expected = Data::from([
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_bool_repeat_ops() {
        let tensor = Tensor::<TestBackend, 2, Bool>::from([[true, false, false]]);

        let data_actual = tensor.repeat(0, 4).into_data();

        let data_expected = Data::from([
            [true, false, false],
            [true, false, false],
            [true, false, false],
            [true, false, false],
        ]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_int_repeat_ops() {
        let tensor = Tensor::<TestBackend, 2, Int>::from([[0, 1, 2]]);

        let data_actual = tensor.repeat(0, 4).into_data();

        let data_expected = Data::from([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]);
        assert_eq!(data_expected, data_actual);
    }
}
