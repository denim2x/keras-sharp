using System.Collections.Generic;
using System.Linq;

namespace KerasSharp.Engine.Topology {
    public abstract class Cell {
        public virtual int?[] state_size { get { return null; } }

        public List<Tensor> this[params Tensor[] x] {
            get { return Call(x.ToList()); }
        }

        ///  <summary>
        ///    Wrapper around this.call(), for handling internal references.
        ///  </summary>
        ///  
        ///  <remarks>
        ///  If a Keras tensor is passed:
        ///  -We call this._add_inbound_node().
        ///  -If necessary, we `build` the layer to match the _keras_shape of the input(s).
        ///  -We update the _keras_shape of every input tensor with its new shape(obtained via this.compute_output_shape). This is done as part of _add_inbound_node().
        ///  -We update the _keras_history of the output tensor(s) with the current layer. This is done as part of _add_inbound_node().
        ///  </remarks>
        ///  
        ///  <param name="inputs">Can be a tensor or list/ tuple of tensors.</param>
        ///  
        ///  <returns>Output of the layer"s `call` method.</returns>.
        ///  
        public virtual List<Tensor> Call(Tensor input, Tensor mask = null, bool? training = null) {
            if (mask == null)
                return Call(new List<Tensor> { input }, null, training);
            return Call(new List<Tensor> { input }, new List<Tensor> { mask }, training);
        }

        ///  <summary>
        ///    Wrapper around this.call(), for handling internal references.
        ///  </summary>
        ///  
        ///  <remarks>
        ///  If a Keras tensor is passed:
        ///  -We call this._add_inbound_node().
        ///  -If necessary, we `build` the layer to match the _keras_shape of the input(s).
        ///  -We update the _keras_shape of every input tensor with its new shape(obtained via this.compute_output_shape). This is done as part of _add_inbound_node().
        ///  -We update the _keras_history of the output tensor(s) with the current layer. This is done as part of _add_inbound_node().
        ///  </remarks>
        ///  
        ///  <param name="inputs">Can be a tensor or list/ tuple of tensors.</param>
        ///  
        ///  <returns>Output of the layer"s `call` method.</returns>.
        ///  
        public virtual List<Tensor> Call(List<Tensor> inputs, List<Tensor> mask = null, bool? training = null) {
            return inputs;
        }
    }
}
