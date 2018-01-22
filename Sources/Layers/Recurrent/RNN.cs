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
  using static System.Linq.Enumerable;
  using Shapes = System.Collections.Generic.List<int?[]>;

  using System.Runtime.Serialization;
  using KerasSharp.Constraints;
  using KerasSharp.Regularizers;
  using KerasSharp.Initializers;
  using Accord.Math;
  using KerasSharp.Engine.Topology;
  using KerasSharp.Activations;
  using KerasSharp.Utils;

  using static KerasSharp.Backends.Current;

  /// <summary>
  /// Base class for recurrent layers.
  /// 
  /// Based on https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1378
  /// </summary>
  [DataContract]
  public class RNN : Layer {
    public Cell cell;
    public bool return_sequences;
    public bool return_state;
    public bool go_backwards;
    public bool unroll;

    public List<InputSpec> state_spec;
    public int num_states { get; protected set; }
    protected List<Tensor> _states;

    public List<InputSpec> constants_spec;
    public int? _num_constants { get; protected set; }

    public RNN(
      Cell cell,
      bool return_sequences = false,
      bool return_state = false,
      bool go_backwards = false,
      bool stateful = false,
      bool unroll = false)
      : base()
    {
      this.cell = cell;
      this.return_sequences = return_sequences;
      this.return_state = return_state;
      this.go_backwards = go_backwards;
      this.stateful = stateful;
      this.unroll = unroll;

      supports_masking = true;
      input_spec = new List<InputSpec> {
        new InputSpec(ndim: 3)
      };
      state_spec = null;
      _states = null;
      constants_spec = null;
      _num_constants = null;
    }

    public List<Tensor> states {
      get {
        if (_states == null) {
          num_states = cell.state_size.Length;
          return Repeat<Tensor>(null, num_states).ToList();
        }
        return _states;
      }
      set {
        _states = value;
      }
    }

    public override Shapes compute_output_shape(List<int?[]> input_shape) {
      var state_size = cell.state_size;
      var output_dim = state_size[0];
      var inputShape = input_shape[0];

      int?[] outputShape;
      if (return_sequences) {
        outputShape = new []{ inputShape[0], inputShape[1], output_dim };
      } else {
        outputShape = new []{ inputShape[0], output_dim };
      }

      var output_shape = new Shapes{ outputShape };
      if (return_state) {
        output_shape.AddRange(state_size.Select((dim) => new[] { inputShape[0], dim }));
      }

      return output_shape;
    }

    public override List<Tensor> compute_mask(List<Tensor> inputs, List<Tensor> mask) {
      var _mask = mask[0];
      var output_mask = new List<Tensor>{ return_sequences ? _mask : null };
      if (return_state) {
        output_mask.AddRange(Repeat<Tensor>(null, states.Count));
      }
      return output_mask;
    }

    protected override void build(Shapes input_shape) {
      Shapes constants_shape;
      if (_num_constants == null) {
        constants_shape = null; 
      } else {
        constants_shape = input_shape.skip(-_num_constants.Value).ToList();
      }

      var inputShape = input_shape[0];
      var batch_size = stateful ? inputShape[0] : null;
      var input_dim = inputShape.get(-1);
      input_spec[0] = new InputSpec(shape: new int?[]{ batch_size, null, input_dim });


    }
  }
}
