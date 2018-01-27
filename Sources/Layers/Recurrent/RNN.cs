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
    public class RNN<TCell> : Layer
        where TCell : Cell {
        public TCell cell;
        public bool return_sequences = false;
        public bool return_state = false;
        public bool go_backwards = false;
        public bool unroll = false;

        public List<InputSpec> state_spec { get; protected set; } = null;
        public int num_states { get; protected set; }
        public List<Tensor> _states { get; protected set; } = null;

        public List<InputSpec> constants_spec = null;
        public int? _num_constants { get; protected set; } = null;

        public RNN() {
            stateful = false;
            supports_masking = true;
            input_spec = new List<InputSpec> {
                new InputSpec(ndim: 3)
            };
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

        public override Shapes compute_output_shape(Shapes input_shape) {
            var state_size = cell.state_size;
            var output_dim = state_size[0];
            var inputShape = input_shape[0];

            int?[] outputShape;
            if (return_sequences) {
                outputShape = new[] { inputShape[0], inputShape[1], output_dim };
            } else {
                outputShape = new[] { inputShape[0], output_dim };
            }

            var output_shape = new Shapes { outputShape };
            if (return_state) {
                output_shape.AddRange(state_size.Select((dim) => new[] { inputShape[0], dim }));
            }

            return output_shape;
        }

        public override List<Tensor> compute_mask(List<Tensor> inputs, List<Tensor> mask) {
            var _mask = mask[0];
            var output_mask = new List<Tensor> { return_sequences ? _mask : null };
            if (return_state) {
                output_mask.AddRange(Repeat<Tensor>(null, states.Count));
            }
            return output_mask;
        }

        internal override void build(Shapes input_shape) {
            Shapes constants_shape;
            if (_num_constants == null) {
                constants_shape = null;
            } else {
                constants_shape = input_shape.skip(-_num_constants.Value).ToList();
            }

            var inputShape = input_shape[0];
            var batch_size = stateful ? inputShape[0] : null;
            var input_dim = inputShape.get(-1);
            input_spec[0] = new InputSpec(shape: new[] { batch_size, null, input_dim });

            if (cell is Layer layer) {
                var step_input_shape = new[] { inputShape[0] }.Concat(inputShape.skip(2)).ToArray();
                if (constants_shape == null) {
                    layer.build(new Shapes { step_input_shape });
                } else {
                    layer.build(new[] { step_input_shape }.Concat(constants_shape).ToList());
                }
            }

            var state_size = cell.state_size;
            if (state_spec == null) {
                state_spec = state_size.Select((dim) => new InputSpec(shape: new[] { null, dim })).ToList();
            } else if (state_spec.any(state_size, (spec, size) => spec.shape.get(-1) != size)) {
                throw new InvalidOperationException($@"An `initial_state` was passed that is not compatible 
with `cell.state_size`. Received `state_spec`=${state_spec}; however `cell.state_size` is ${cell.state_size}");
            }

            if (stateful) {
                reset_states();
            }

            base.build(input_shape);
        }


    }
}
