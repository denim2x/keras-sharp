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

    using System.Runtime.Serialization;
    using KerasSharp.Constraints;
    using KerasSharp.Regularizers;
    using KerasSharp.Initializers;
    using Accord.Math;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Activations;
    using KerasSharp.Utils;

    using static KerasSharp.Backends.Current;
    using Shapes = System.Collections.Generic.List<int?[]>;
    using Tensors = System.Collections.Generic.List<Engine.Topology.Tensor>;
    using InputSpecs = System.Collections.Generic.List<InputSpec>;

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
        public Tensors _states { get; protected set; } = null;

        public List<InputSpec> constants_spec = null;
        public int? _num_constants { get; protected set; } = null;

        public RNN() {
            stateful = false;
            supports_masking = true;
            input_spec = new List<InputSpec> {
                new InputSpec(ndim: 3)
            };
        }

        public Tensors states {
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

        public override Tensors compute_mask(Tensors inputs, Tensors mask) {
            var _mask = mask[0];
            var output_mask = new Tensors { return_sequences ? _mask : null };
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
            } else if (state_spec.any(state_size, (spec, size) => spec.shape.get(-1) != size, true)) {
                throw new Errors.Value(Wrappers.square, 
                    @"An `initial_state` was passed that is not compatible with `cell.state_size`;
received `state_spec`={0}, whereas `cell.state_size`={1}.",
                    state_spec, cell.state_size);
            }

            if (stateful) {
                reset_states();
            }

            base.build(input_shape);
        }

        public Tensors get_initial_state(Tensor inputs) {
            var initial_state = K.expand_dims(K.sum(K.zeros_like(inputs), axis: new[] { 1, 2 }));
            return cell.state_size.Select((dim) => K.tile(initial_state, new[] { 1, dim })).ToList();
        }

        public Tensors Call(Tensor inputs, Tensors initial_state = null, Tensors constants = null) {
            if (initial_state == null && constants == null) {
                return base.Call(inputs);
            }

            var _inputs = new Tensors();
            var _specs = new InputSpecs();

            if (initial_state != null) {
                _inputs.AddRange(initial_state);
                state_spec = initial_state.Select((state) => new InputSpec(shape: K.int_shape(state))).ToList();
                _specs.AddRange(state_spec);
            }

            if (constants != null) {
                _inputs.AddRange(constants);
                constants_spec = constants.Select((constant) => new InputSpec(shape: K.int_shape(constant))).ToList();
                _num_constants = constants.Count;
                _specs.AddRange(constants_spec);
            }

            var input = _inputs.Prepend(inputs).ToList();
            var input_spec = this.input_spec.Concat(_specs).ToList();
            var _input_spec = this.input_spec;
            this.input_spec = input_spec;
            var output = base.Call(input);
            this.input_spec = _input_spec;
            return output;
        }

        public List<Tensor> Call(Tensor inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null) {
            if (initial_state == null && stateful) {
                initial_state = states;
            } else {
                initial_state = get_initial_state(inputs);
            }

            if (initial_state.Count != states.Count) {
                throw new Errors.Value(Wrappers.square, 
                    "Layer has {0} states but was passed {1} initial states.", 
                    states.Count, initial_state.Count);
            }

            var input_shape = K.int_shape(inputs);
            var timesteps = input_shape[1];
            if (unroll && _timesteps.Contains(timesteps)) {
                throw new Errors.Value(
                    (@"Cannot unroll a RNN if the time dimension is undefined or equal to 1.",
                    @"- If using a Sequential model, specify the time dimension by passing an `input_shape` or
`batch_input_shape` argument to your first layer. If your first layer is an Embedding, you can also use the
`input_length` argument.",
                    @"- If using the functional API, specify the time dimension by passing a `shape` or `batch_shape`
argument to your Input layer."
                    ));
            }

            if (constants != null) {

            }
        }

        internal readonly HashSet<int?> _timesteps = new HashSet<int?> { null, 1 };

        public List<Tensor> Call(Tensors inputs, Tensors mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null) {
            return Call(inputs[0], mask?[0], training, initial_state, constants);
        }
    }
}
