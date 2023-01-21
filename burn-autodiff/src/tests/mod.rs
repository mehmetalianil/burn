mod add;
mod aggregation;
mod backward;
mod cat;
mod complex;
mod conv1d;
mod conv2d;
mod cross_entropy;
mod div;
mod erf;
mod exp;
mod index;
mod log;
mod mask;
mod matmul;
mod maxpool2d;
mod mul;
mod multithread;
mod neg;
mod pow;
mod relu;
mod reshape;
mod softmax;
mod sqrt;
mod sub;
mod transpose;

#[macro_export]
macro_rules! testgen_all {
    () => {
        type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;
        type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        type TestADTensor<const D: usize> = burn_tensor::Tensor<TestADBackend, D>;

        // Modules
        burn_autodiff::testgen_ad_conv1d!();
        burn_autodiff::testgen_ad_conv2d!();
        burn_autodiff::testgen_ad_max_pool2d!();
        burn_autodiff::testgen_module_backward!();

        // Tensor
        burn_autodiff::testgen_ad_complex!();
        burn_autodiff::testgen_ad_multithread!();
        burn_autodiff::testgen_ad_add!();
        burn_autodiff::testgen_ad_aggregation!();
        burn_autodiff::testgen_ad_cat!();
        burn_autodiff::testgen_ad_cross_entropy_loss!();
        burn_autodiff::testgen_ad_div!();
        burn_autodiff::testgen_ad_erf!();
        burn_autodiff::testgen_ad_exp!();
        burn_autodiff::testgen_ad_index!();
        burn_autodiff::testgen_ad_log!();
        burn_autodiff::testgen_ad_mask!();
        burn_autodiff::testgen_ad_matmul!();
        burn_autodiff::testgen_ad_mul!();
        burn_autodiff::testgen_ad_neg!();
        burn_autodiff::testgen_ad_powf!();
        burn_autodiff::testgen_ad_sqrt!();
        burn_autodiff::testgen_ad_relu!();
        burn_autodiff::testgen_ad_reshape!();
        burn_autodiff::testgen_ad_softmax!();
        burn_autodiff::testgen_ad_sub!();
        burn_autodiff::testgen_ad_transpose!();
    };
}
