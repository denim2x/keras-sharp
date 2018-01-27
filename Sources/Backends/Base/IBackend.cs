﻿// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

namespace KerasSharp.Backends {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Losses;
    using KerasSharp.Models;

    public interface IBackend : IDisposable {
        // TODO: Rename all methods to camelCase

        Tensor sqrt(Tensor x);

        Tensor square(Tensor w);
        Tensor equal(Tensor x, Tensor y);
        Tensor sum(Tensor x, int[] axis = null, bool keepdims = false, string name = null);
        Tensor sum(List<Tensor> x, int[] axis = null, bool keepdims = false, string name = null);
        Tensor round(Tensor x);
        Tensor argmax(Tensor x, int axis = -1);
        Tensor sum(Tensor x, int axis, bool keepdims = false, string name = null);
        Tensor batch_flatten(Tensor inputs);


        Tensor clip(Tensor norms, int v, int maxValue);

        Tensor zeros(int[] shape, DataType? dtype = null, string name = null);

        Tensor zeros(int?[] shape, DataType? dtype = null, string name = null);

        Tensor zeros_like(Tensor inputs);

        Tensor expand_dims(Tensor inputs);

        Tensor tile(Tensor inputs, int?[] axis);

        float epsilon();

        DataType floatx();

        Tensor greater_equal(Tensor w, double v);
        void clear_session();

        Tensor cast(Tensor x, DataType dataType);


        Tensor dropout(object p, double retain_prob, object noise_shape, object seed);

        Tensor relu(Tensor x);

        Tensor softmax(Tensor x);


        Tensor max(Tensor x, int v, object p);

        Tensor reshape(Tensor x, int[] shape);

        int? ndim(Tensor x);

        Tensor max(Tensor x, int axis, bool keepdims);


        Tensor elu(Tensor x);

        Tensor hard_sigmoid(Tensor x);


        Tensor mul(Tensor a, Tensor b, string name = null);

        Tensor pow(Tensor x, Tensor p, string name = null);

        Tensor mul<T>(T a, Tensor b, string name = null);

        Tensor mul<T>(Tensor a, T b, string name = null);

        Tensor mul(List<Tensor> batch_outs, int length);




        Tensor div(Tensor a, Tensor b);

        Tensor div<T>(T a, Tensor b);

        Tensor div<T>(Tensor a, T b);





        Tensor add(Tensor a, Tensor b);

        Tensor add<T>(Tensor a, T b);

        Tensor add<T>(T a, Tensor b);


        Tensor subtract(Tensor a, Tensor b, string name = null);

        Tensor subtract<T>(Tensor a, T b, string name = null);


        Tensor subtract<T>(T a, Tensor b, string name = null);



        Tensor dot(Tensor a, Tensor b, string name = null);


        Tensor elu(Tensor x, double alpha);

        Tensor sigmoid(Tensor x);

        Tensor softplus(Tensor x);

        Tensor print_tensor(Tensor x, string message);

        DataFormatType image_data_format();

        Tensor softsign(Tensor x);

        Tensor tanh(Tensor x);

        Tensor exp(Tensor x);

        object eval(Tensor tensor);




        Tensor clip(Tensor norms, double min_value, double max_value);



        Tensor random_uniform(int[] shape, double minval = 0.0, double maxval = 1.0, DataType? dtype = null, int? seed = null, string name = null);

        Tensor l2_normalize(Tensor expected, int axis);

        Tensor minus(Tensor tensor);

        Tensor mean(Tensor tensor, int axis = -1, bool keepdims = false, string name = null);

        Tensor mean(Tensor tensor, int[] axis, bool keepdims = false, string name = null);

        Tensor abs(Tensor input);

        Tensor categorical_crossentropy(Tensor target, Tensor output, bool from_logits = false);

        Tensor max(Tensor tensor, int axis);

        Tensor maximum(double v, Tensor tensor);

        Tensor binary_crossentropy(Tensor output, Tensor target, bool from_logits = false);




        Tensor variable(Array array, DataType? dtype = null, string name = null);

        Tensor variable<T>(T value, DataType? dtype = null, string name = null) where T : struct;

        Tensor variable(Tensor tensor, DataType? dtype = null, string name = null);

        Tensor in_train_phase(Func<Tensor> x, Func<Tensor> alt, bool? training);

        DataType? dtype(Tensor input_tensor);

        Tensor constant<T>(T value, int[] shape = null, DataType? dtype = null, string name = null);

        Tensor transpose(Tensor tensor);

        int get_uid(string prefix);

        List<Tensor> gradients(Tensor loss, List<Tensor> param);

        int?[] int_shape(Tensor tensor);


        NameScope name_scope(string name);

        Tensor clip_norm(Tensor g, double clipnorm, Tensor norm);

        Tensor identity(Tensor x, string name = null);

        List<Array> batch_get_value(List<Tensor> weights);

        void batch_set_value(List<Tuple<Tensor, Array>> weight_value_tuples);

        Tensor placeholder(int?[] shape = null, int? ndim = null, DataType? dtype = null, bool sparse = false, string name = null);

        // Tensor placeholder(int ndim, TFDataType? dtype = Utils.DEFAULT_DTYPE, bool sparse = false, string name = null);

        List<Array> batch_get_value(List<List<Tensor>> weights);

        void batch_set_value(List<(Tensor, Array)> tuples);

        Tensor update_add<T>(Tensor x, T increment, string name = null) where T : struct;


        int?[] get_variable_shape(Tensor x);

        Tensor sum(double v, Tensor tensor);

        bool is_sparse(Tensor tensor);


        object learning_phase();

        Function function(List<Tensor> inputs, List<Tensor> list, List<List<Tensor>> updates, string name);

        Tensor update(Tensor x, Tensor new_x, string name = null);

        Tensor truncated_normal(int[] shape, double v, double stddev, DataType? dtype, int? seed);

        Tensor truncated_normal(int?[] shape, double v, double stddev, DataType? dtype, int? seed);

        Tensor not_equal<T>(Tensor weights, T v) where T : struct;

        Tensor bias_add(Tensor output, Tensor bias, DataFormatType? data_format = null, string name = null);

        Tensor conv1d(Tensor inputs, Tensor kernel, int strides, PaddingType padding, DataFormatType? data_format, int dilation_rate, string name = null);

        Tensor conv2d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format, int[] dilation_rate, string name = null);

        Tensor conv3d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format, int[] dilation_rate, string name = null);
    }
}
