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
  using System;
  using System.Collections.Generic;
  using System.Linq;
  using System.Text;
  using System.Threading.Tasks;

  using System.Runtime.Serialization;
  using KerasSharp.Constraints;
  using KerasSharp.Regularizers;
  using KerasSharp.Initializers;
  using Accord.Math;
  using KerasSharp.Engine.Topology;
  using KerasSharp.Activations;

  using static KerasSharp.Backends.Current;

  /// <summary>
  /// Gated Recurrent Unit - Cho et al. 2014.
  /// 
  /// Based on https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1378
  /// </summary>
  [DataContract]
  public class GRU : RNN {
    public GRU(
      IActivationFunction activation = null,
      IActivationFunction recurrent_activation = null,
      bool use_bias = true, 
      IWeightInitializer kernel_initializer = null,
      IWeightInitializer recurrent_initializer = null,
      IWeightInitializer bias_initializer = null,
      IWeightRegularizer kernel_regularizer = null,
      IWeightRegularizer recurrent_regularizer = null,
      IWeightRegularizer bias_regularizer = null,
      IWeightRegularizer activity_regularizer = null,
      IWeightConstraint kernel_constraint = null,
      IWeightConstraint recurrent_constraint = null,
      IWeightConstraint bias_constraint = null,
      double dropout = 0.0,
      double recurrent_dropout,
      bool return_sequences = false,
      bool return_state = false,
      bool go_backwards = false,
      bool stateful = false,
      bool unroll = false) 
      :base()
    {
      if (activation == null) {
        activation = new TanH();
      }
      if (recurrent_activation == null) {
        recurrent_activation = new HardSigmoid();
      }

    }
  }
}
