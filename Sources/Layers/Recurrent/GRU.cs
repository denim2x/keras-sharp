// Keras-Sharp: C# port of the Keras library
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

namespace KerasSharp {
    using System.Runtime.Serialization;
    using KerasSharp.Constraints;
    using KerasSharp.Regularizers;
    using KerasSharp.Initializers;
    using Accord.Math;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Activations;

    using static KerasSharp.Backends.Current;
    using Activator = IActivationFunction;
    using Initializer = Initializers.IWeightInitializer;
    using Regularizer = Regularizers.IWeightRegularizer;
    using Constraint = Constraints.IWeightConstraint;

    /// <summary>
    /// Gated Recurrent Unit - Cho et al. 2014.
    /// 
    /// Based on https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1378
    /// </summary>
    [DataContract]
    public class GRU : RNN<GRUCell> {
        public GRU() {
            cell = new GRUCell();
            activity_regularizer = null;
        }

        public int units {
            get => cell.units;
            set => cell.units = value;
        }

        public Activator activation {
            get => cell.activation;
            set => cell.activation = value;
        }

        public Activator recurrent_activation {
            get => cell.recurrent_activation;
            set => cell.recurrent_activation = value;
        }

        public bool use_bias {
            get => cell.use_bias;
            set => cell.use_bias = value;
        }

        public Initializer kernel_initializer {
            get => cell.kernel_initializer;
            set => cell.kernel_initializer = value;
        }

        public Initializer recurrent_initializer {
            get => cell.recurrent_initializer;
            set => cell.recurrent_initializer = value;
        }

        public Initializer bias_initializer {
            get => cell.bias_initializer;
            set => cell.bias_initializer = value;
        }

        public Regularizer kernel_regularizer {
            get => cell.kernel_regularizer;
            set => cell.kernel_regularizer = value;
        }

        public Regularizer recurrent_regularizer {
            get => cell.recurrent_regularizer;
            set => cell.recurrent_regularizer = value;
        }

        public Regularizer bias_regularizer {
            get => cell.bias_regularizer;
            set => cell.bias_regularizer = value;
        }

        public Constraint kernel_constraint {
            get => cell.kernel_constraint;
            set => cell.kernel_constraint = value;
        }

        public Constraint recurrent_constraint {
            get => cell.recurrent_constraint;
            set => cell.recurrent_constraint = value;
        }

        public Constraint bias_constraint {
            get => cell.bias_constraint;
            set => cell.bias_constraint = value;
        }

        public double dropout {
            get => cell.dropout;
            set => cell.dropout = value;
        }

        public double recurrent_dropout {
            get => cell.recurrent_dropout;
            set => cell.recurrent_dropout = value;
        }
    }
}
