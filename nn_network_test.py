from manimlib.imports import *
import numpy as np

class Network(object):
    def __init__(self, sizes, non_linearity = "sigmoid"):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        if non_linearity == "sigmoid":
            self.non_linearity = sigmoid
            self.d_non_linearity = sigmoid_prime
        elif non_linearity == "ReLU":
            self.non_linearity = ReLU
            self.d_non_linearity = ReLU_prime
        else:
            raise Exception("Invalid non_linearity")

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
    result = np.array(z)
    result[result < 0] = 0
    return result

def ReLU_prime(z):
    return (np.array(z) > 0).astype('int')

class NetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius" : 0.15,
        "neuron_to_neuron_buff" : 3,
        "layer_to_layer_buff" : LARGE_BUFF,
        "neuron_stroke_color" : BLUE,
        "neuron_stroke_width" : 3,
        "neuron_fill_color" : GREEN,
        "edge_color" : LIGHT_GREY,
        "edge_stroke_width" : 2,
        "edge_propogation_color" : YELLOW,
        "edge_propogation_time" : 1,
        "max_shown_neurons" : 16,
        "brace_for_large_layers" : True,
        "average_shown_activation_of_large_layer" : True,
        "include_output_labels" : False,
    }
    def __init__(self, neural_network, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.neural_network = neural_network
        self.layer_sizes = neural_network.sizes
        self.add_neurons()
        self.add_edges()
        self.add_bias()
        self.add_bias_edges()
        self.add_outputs()
        self.add_output_edges()
        #self.add_arrows()
        

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.layer_sizes
        ])

        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers)

        if self.include_output_labels:
            self.add_output_labels()

    def add_outputs(self):
        outputs = VGroup(*[
            self.get_layer(self.layer_sizes[-1]),
            self.get_layer(size=1)
        ])
        #outputs.arrange(RIGHT, buff=self.layer_to_layer_buff)
        outputs.move_to(self.layers[-1])
        # add 2*self.neuron_radius to make spacing equal
        outputs.shift(RIGHT*(self.layer_to_layer_buff))
        outputs[-1].shift(RIGHT*(self.layer_to_layer_buff))

        self.outputs = outputs
        self.add(self.outputs)

    def add_bias(self):
        # add bias layer neurons
        bias = VGroup(*[
            self.get_layer(1) for layer in self.layer_sizes[1:]
            ])

        bias.arrange(RIGHT, buff = self.layer_to_layer_buff)
        bias.move_to(self.layers[1])
        # add 2*self.neuron_radius to make spacing equal
        bias.shift(DOWN*1.1*(self.neuron_to_neuron_buff))
        # Add the bias layer neuron to the network layer neuron list
        # this will help add edges using add_edges (maybe...)
        #[self.layers[i].neurons.add(b.neurons) for i,b in enumerate(bias)]

        self.bias = bias
        self.add(self.bias)

    def expand_nodes(self, layer_index):
        new_nodes = self.get_layer(self.layer_sizes[layer_index])
        new_nodes.move_to(self.layers[layer_index])
        #new_nodes.shift(RIGHT*0.5*(self.layer_to_layer_buff))        
        self.new_nodes = new_nodes

        return(new_nodes)
        
    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius = self.neuron_radius,
                stroke_color = self.neuron_stroke_color,
                stroke_width = self.neuron_stroke_width,
                fill_color = self.neuron_fill_color,
                fill_opacity = 0,
            )
            for x in range(n_neurons)
        ])   
        neurons.arrange(
            DOWN, buff = self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexMobject("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self, output=False):

        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def add_bias_edges(self):
        # Assumes edges run first and self.edge_groups exists.
        self.bias_edge_groups = VGroup()
        for l1, l2 in zip(self.bias, self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge.set_opacity(0.3)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.bias_edge_groups.add(edge_group)
        self.add_to_back(self.bias_edge_groups)
    
    def add_output_edges(self):
        self.out_edge_groups = VGroup()
        out_edge_group = VGroup()
        # Add edges from the last layer to the error nodes
        for n1, n2 in zip(self.layers[-1].neurons, self.outputs[0].neurons):
            edge = self.get_edge(n1, n2)
            n1.edges_out.add(edge)
            n2.edges_in.add(edge)
            n1.next = VGroup()
            n1.next.add(n2)
            out_edge_group.add(edge)

        self.out_edge_groups.add(out_edge_group)

        # Add edges from the error node to the Error Total
        error_edge_group = VGroup()
        for n1, n2 in zip(self.outputs[0].neurons, it.cycle(self.outputs[-1].neurons)):
            edge = self.get_edge(n1, n2)
            n1.edges_out.add(edge)
            error_edge_group.add(edge)

        self.out_edge_groups.add(error_edge_group)
        self.add_to_back(self.out_edge_groups)

    def get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff = self.neuron_radius,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width,
        )

    def add_arrows(self):
        self.arrow_groups = VGroup()
        for n1, n2 in zip(self.layers[-1].neurons, self.outputs[0].neurons):
            arrow = self.get_arrow(n1, n2)
            self.arrow_groups.add(arrow)
            #n1.edges_out.add(edge)
            #n2.edges_in.add(edge)
        self.add_to_back(self.arrow_groups)


    def get_arrow(self, neuron1, neuron2):
        return Arrow(
            neuron1.get_center(),
            neuron2.get_center(),
            buff = self.neuron_radius,
            stroke_color = BLUE,
            #stroke_width = self.edge_stroke_width,
        )

    def get_active_layer(self, layer_index, activation_vector):
        layer = self.layers[layer_index].deepcopy()
        self.activate_layer(layer, activation_vector)
        return layer

    def activate_layer(self, layer, activation_vector):
        n_neurons = len(layer.neurons)
        av = activation_vector
        def arr_to_num(arr):
            return (np.sum(arr > 0.1) / float(len(arr)))**(1./3)

        if len(av) > n_neurons:
            if self.average_shown_activation_of_large_layer:
                indices = np.arange(n_neurons)
                indices *= int(len(av)/n_neurons)
                indices = list(indices)
                indices.append(len(av))
                av = np.array([
                    arr_to_num(av[i1:i2])
                    for i1, i2 in zip(indices[:-1], indices[1:])
                ])
            else:
                av = np.append(
                    av[:n_neurons/2],
                    av[-n_neurons/2:],
                )
        for activation, neuron in zip(av, layer.neurons):
            neuron.set_fill(
                color = self.neuron_fill_color,
                opacity = activation
            )
        return layer

    def activate_layers(self, input_vector):
        activations = self.neural_network.get_activation_of_all_layers(input_vector)
        for activation, layer in zip(activations, self.layers):
            self.activate_layer(layer, activation)

    def deactivate_layers(self):
        all_neurons = VGroup(*it.chain(*[
            layer.neurons
            for layer in self.layers
        ]))
        all_neurons.set_fill(opacity = 0)
        return self

    def get_edge_propogation_animations(self, index, bias=False):
        if bias:
           edge_group_copy = self.bias_edge_groups[index].copy() 
        else:
            edge_group_copy = self.edge_groups[index].copy()
        
        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width = 1.5*self.edge_stroke_width
        )
        return [ShowCreationThenDestruction(
            edge_group_copy, 
            run_time = self.edge_propogation_time,
            lag_ratio = 0.5
        )]

    # def add_output_labels(self):
    #     self.output_labels = VGroup()
    #     for n, neuron in enumerate(self.layers[-1].neurons):
    #         label = TexMobject(str(n))
    #         label.set_height(0.75*neuron.get_height())
    #         label.move_to(neuron)
    #         label.shift(neuron.get_width()*RIGHT)
    #         self.output_labels.add(label)
    #     self.add(self.output_labels)


class NetworkScene(Scene):
    CONFIG = {
        "layer_sizes" : [2, 2, 2],
        "label_scale" : 0.75,
        "network_mob_config" : {},
    }

    # The Scene class __init__ method calls this setup method
    def setup(self):
        self.add_network()

    def add_network(self):
        self.network = Network(sizes = self.layer_sizes)
        self.network_mob = NetworkMobject(
            self.network,
            **self.network_mob_config
        )
        self.add(self.network_mob)

    def feed_forward(self, input_vector, false_confidence = False, added_anims = None):
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        for i, activation in enumerate(activations):
            self.show_activation_of_layer(i, activation, added_anims)
            added_anims = []

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims = None):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector
        )
        anims = [Transform(layer, active_layer)]
        if layer_index > 0:
            anims += self.network_mob.get_edge_propogation_animations(
                layer_index-1
            )
        anims += added_anims
        self.play(*anims)

    def remove_random_edges(self, prop = 0.9):
        for edge_group in self.network_mob.edge_groups:
            for edge in list(edge_group):
                if np.random.random() < prop:
                    edge_group.remove(edge)

def make_transparent(image_mob):
    alpha_vect = np.array(
        image_mob.pixel_array[:,:,0],
        dtype = 'uint8'
    )
    image_mob.set_color(WHITE)
    image_mob.pixel_array[:,:,3] = alpha_vect
    return image_mob

class NetworkSetup(NetworkScene):
    CONFIG = {
        "network_mob_config" : {
            "neuron_stroke_color" : WHITE,
            "neuron_fill_color" : WHITE,
            "neuron_radius" : 0.45,
            "layer_to_layer_buff" : 2,
        },
        "label_scale" : 0.75,
        "layer_sizes" : [2, 2, 2],
    }
    def construct(self):
        self.setup_network_mob()
        self.show_labels()
        self.show_weights()
        #self.reposition_error()
        self.play(FadeOut(self.error_sum))
        self.insert_activations(layer_index=2)
        self.insert_activations(layer_index=1)
        self.rescale_network()
        layer = 2
        weight_indexs_2 = [[layer,1,1], [layer,1,2], [layer,2,2], [layer,2,1]]
        directions = [UP,3.5*RIGHT, DOWN, 3.5*LEFT]
        for weight_index, direction in zip(weight_indexs_2, directions):
            self.find_derivative(weight_index)
            #self.trace_back(weight_index=weight_index)
            self.partial_derivative(weight_index=weight_index)
            self.step_through(weight_index)
            self.make_room(direction)
        self.play(*self.show_col_matrix(self.de_da_terms.copy(), BLUE_B))
        self.play(*self.show_col_matrix(self.da_dz_terms.copy(), YELLOW_B))
        self.play(*self.show_col_matrix(self.dz_dw_terms.copy(), RED_B, matrix=True))
        self.add_multipliers()
        self.write_out_formulas()
        self.reset_network()
        
        weight_index_1 = [1,1,1]
        self.find_derivative(weight_index_1)
        self.partial_derivative(weight_index=weight_index_1)
        self.step_through(weight_index_1)
        #self.make_room(0.01*DOWN)
        self.recall_delta()
        layer_1 = 1
        weight_indexs_1 = [[layer_1,1,2], [layer_1,2,2], [layer_1,2,1]]
        directions_1 = [3.5*RIGHT, DOWN, 3.5*LEFT]
        for i, (weight_index, direction) in enumerate(zip(weight_indexs_1, directions_1)):
            if i == 2:
                self.play(self.equation[-1].fade, 0.8)

            self.find_derivative(weight_index)
            self.partial_derivative(weight_index=weight_index)
            self.step_through(weight_index)
            self.delta_transform()
            self.make_room(direction)
        self.play(self.equation[-2].set_opacity, 1)

        self.play(FadeOut(self.full_network))
        self.play(*self.show_col_matrix(self.de_da_terms[-4:].copy(), BLUE_B, position=DOWN))
        self.play(*self.show_col_matrix(self.da_dz_terms[-4:].copy(), BLUE_B))

        #self.play(FadeOut(self.equation[-2]))
        #self.show_bias()
        #self.organize_activations_into_column()
        #self.organize_weights_as_matrix()
        #self.show_meaning_of_matrix_row()
        #self.connect_weighted_sum_to_matrix_multiplication()
        #self.add_bias_vector()
        #self.apply_sigmoid()
        #self.write_clean_final_expression()

    def setup_network_mob(self):
        self.network_mob.to_edge(LEFT, buff = LARGE_BUFF)
        #self.network_mob.layers[1].neurons.shift(0.02*RIGHT)

    def show_labels(self):
        #self.fade_many_neurons()
        self.activate_layer(0)
        self.activate_layer(1)
        self.activate_layer(2)
        self.activate_output()
        neuron_1 = 0
        neuron_2 = 1
        self.show_error(neuron_1, UP, 'bce')
        self.show_error(neuron_2, DOWN, 'mse')
        #self.show_first_neuron_weighted_sum()
        #self.add_bias()
        #self.add_sigmoid()
    ##

    def fade_many_neurons(self):
        anims = []
        neurons = self.network_mob.layers[1].neurons
        for neuron in neurons[1:]:
             # Saved_state is just a copy() method and allows you to use 
             # the restore method of the mobject class to unfade later on.
            neuron.save_state()  
            neuron.edges_in.save_state()
            anims += [
                neuron.fade, 0.8,
                neuron.set_fill, None, 0,
                neuron.edges_in.fade, 0.8,
            ]
        # Not sure why this needs to be added, not rquired thus far...
        # anims += [
        #     Animation(neurons[0]),
        #     Animation(neurons[0].edges_in),
        # ]
        self.play(*anims)


    def activate_layer(self, layer_index, label='a', bias=True, sublayer=False):
        if sublayer:
            layer = self.network_mob.layers[layer_index].sublayer
        else:
            layer = self.network_mob.layers[layer_index]
        #activations = 0.7*np.random.random(len(layer.neurons))
        #active_layer = self.network_mob.get_active_layer(layer_index, activations)

        if layer_index > 0:
            neuron_labels = VGroup(*[
                TexMobject("%s^{(%d)}_%d"%(label, layer_index+1,d+1))
                for d in range(len(layer.neurons))
            ])
            if bias:
                self.activate_bias(layer_index)
        else:
            neuron_labels = VGroup(*[
                TexMobject("x^{(%d)}_%d"%(layer_index+1,d+1))
                for d in range(len(layer.neurons))
            ])

        for label, neuron in zip(neuron_labels, layer.neurons):
            label.scale(0.75)
            label.move_to(neuron)
            neuron.label = label

        #self.play(
         #   Transform(layer, active_layer),
            #Write(neuron_labels, run_time = 1),
        #)
        self.add(neuron_labels)

        if sublayer:
            try: 
                self.sublayer_labels.add_to_back(neuron_labels)
            except:
                #TODO: 
                self.sublayer_labels = VGroup()
                self.sublayer_labels.add_to_back(neuron_labels)
                #self.sublayer_labels = neuron_labels
        else:   
            try: 
                self.neuron_labels.add(neuron_labels)
            except: 
                self.neuron_labels = VGroup()
                self.neuron_labels.add(neuron_labels)

    def activate_output(self):

        layer = self.network_mob.outputs[0]
        #activations = 0.7*np.random.random(len(layer.neurons))
        #active_layer = self.network_mob.get_active_layer(layer_index, activations)

        e_labels = VGroup(*[
            TexMobject("e",
                "^{(%d)}_%d"%(self.network_mob.neural_network.num_layers+1,d+1))
            for d in range(len(layer.neurons))],
            TexMobject("e_{T}")
        )
        e_labels.set_color(ORANGE)
        e_labels.scale(self.label_scale)
        for label, neuron in zip(e_labels[:-1], layer.neurons):
            #label.scale(self.label_scale)
            label.move_to(neuron)
            neuron.label = label
        e_labels[-1].move_to(self.network_mob.outputs[-1].neurons)

        self.play(
            #Transform(layer, active_layer),
            Write(e_labels, run_time = 2)
        )
        self.e_labels = e_labels
    
    def insert_activations(self, layer_index):

        network_mob = self.network_mob
        layer = network_mob.expand_nodes(layer_index)

        self.network_mob.layers[layer_index].sublayer = layer
        
        try:
            sublayer_labels = self.sublayer_labels[layer_index-1:]
        except:
            sublayer_labels=VGroup()

        network_out = VGroup(*[
            network_mob.layers[layer_index:],
            network_mob.edge_groups[layer_index:],
            network_mob.bias[layer_index:],
            self.b_labels[layer_index:],
            network_mob.bias_edge_groups[layer_index:],
            self.bw_labels[layer_index:],
            self.w_labels_nn[layer_index:], 
            self.neuron_labels[layer_index:],
            sublayer_labels[layer_index-1:],
            network_mob.outputs, 
            #network_mob.arrow_groups, 
            network_mob.out_edge_groups,
            self.e_labels
        ])
        
        self.play(
            ApplyMethod(network_out.shift,RIGHT))
        self.play(FadeIn(layer, lag_ratio=0.8))
        
        self.activate_layer(layer_index, label='z', sublayer=True, bias=False)
        #try:
        #    self.network_mob.layers[layer_index].sublayer.add(layer)
        #except:
        self.network_mob.layers[layer_index].add(layer)

    def reposition_error(self):

        error_type = self.error_sum
        error_type.generate_target()
        error_type.target.shift(DOWN)

        self.play(MoveToTarget(error_type))

    def rescale_network(self):
        self.network_scale = 0.7
        network_mob = self.network_mob
        network = VGroup(*[
            network_mob, 
            self.e_labels, 
            self.neuron_labels,
            self.sublayer_labels, 
            self.b_labels,
            self.w_labels_nn,
            self.bw_labels])

        #error_type = self.error_sum
        # error_type.generate_target()
        # error_type.target.arrange(DOWN)
        # error_type.target.to_edge(UR)
        
        network.generate_target()
        network.target.scale_in_place(self.network_scale)
        network.target.to_edge(DOWN)

        self.play(MoveToTarget(network),
                #MoveToTarget(error_type),
            )
        self.full_network = network

    def find_derivative(self, weight_index):
        '''
        weight index is a list [layer, j, k]
        negative layer index's work
        '''   
        layer_index = weight_index[0]
        neuron_2 = weight_index[1]-1
        neuron_1 = weight_index[2]-1

        layer = self.network_mob.layers[layer_index]
        n_neurons = len(layer.neurons)
        L_terms = VGroup() # Penultimate layer terms
        error_t = self.e_labels[-1].copy()
        error_t.generate_target()
        label_index = neuron_1*n_neurons+neuron_2
        target_weight = self.w_labels[layer_index-1][label_index].copy()
        
        network_weight = self.w_labels_nn[layer_index-1][label_index].copy()
        target_weight.move_to(self.w_labels_nn[layer_index-1][0])
        target_weight.shift(0.6*UP)
        error_t.target.scale(1/self.network_scale)
        target_weight.scale(self.label_scale)
        target_weight.set_color(network_weight.get_color())
        error_t.target.align_to(target_weight, DOWN)

        de_dw = TexMobject("{\\partial",error_t.copy().get_tex_string(), 
                        "\\over", 
                        "\\partial", target_weight.copy().get_tex_string(), "}")
        de_dw.scale(self.label_scale)
        de_dw.next_to(self.network_mob, 2*UL)
        de_dw.shift(LEFT)
        arrow = Arrow(error_t.target.get_bottom(), 
            target_weight.get_bottom(), 
            buff=MED_LARGE_BUFF, 
            color=BLUE,
            tip_length=0.3)
        arrow.shift(0.15*UP)

        self.de_dw = VGroup(de_dw)
        self.play(ReplacementTransform(network_weight,target_weight))        
        self.play(MoveToTarget(error_t))
        self.play(GrowArrow(arrow))
        self.trace_back(weight_index)
        self.play(FadeOut(arrow))
        self.play(ReplacementTransform(VGroup(*[target_weight, error_t]), de_dw))

    def trace_back(self, weight_index):
        '''
        weight index is a list [layer, j, k]
        negative layer indexs work
        '''   
        
        def get_edge_propogation_animations():
            edge_group_copy = VGroup(*[n[0].edges_out.copy() for n in self.chosen_neurons[2]])
            flat_list = it.chain.from_iterable(self.chosen_neurons[1])
            # Add edges out for each neuron in the branches
            edge_group_copy.add((*[n.edges_out.copy() for branch in flat_list for n in branch]))
            edge_group_copy.add(self.chosen_neurons[1][0][0].edges_in[neuron_1].copy())
            
            edge_group_copy.set_stroke(
                self.network_mob.edge_propogation_color,
                width = 1.5*self.network_mob.edge_stroke_width
            )
            # A neuron may have multiple edges out so we need to break 
            # this up into individual edges and rotate each seperately so the
            # animated propogation looks form back to front of network
            for edge in edge_group_copy: 
                for e in edge.split():
                    e.rotate_in_place(np.pi) 
            
            return [ShowCreation(
                edge_group_copy, 
                run_time = 2*self.network_mob.edge_propogation_time,
                lag_ratio = 0),
                FadeOut(
                edge_group_copy)]

        network = self.network_mob
        layer_index = weight_index[0]
        neuron_2 = weight_index[1]-1
        neuron_1 = weight_index[2]-1

        chosen_neurons = VGroup()
        layers = network.layers[layer_index:]

        # Add first indexed neuron.  If the desired weight index is
        # deeper in the network there will be multiple dependencies and a 
        # for loop will go through the remaining layers to find these.
        chosen_neurons.add(*[layers[0].sublayer.neurons[neuron_2]]) 
        # create a default branch group, this is nested to keep
        # indexing consistent between single branch and multi
        # branch terms (a branch being a network row that effects the error)
        branches = VGroup(VGroup(layers[0].neurons[neuron_2]))
        # if multiple layers this will find them and overwrite the branches above
        for layer in layers[1:]:
            branches = VGroup() # A branch for each row of neurons 
            for i in range(len(layer.neurons)):
            # Add each branch as a subgroup of the chosen neuron group[1]
            # group[1] comprises all the middle layers and has as many
            # subgroups as branches
                branches.add(VGroup(*[
                    layers[0].neurons[neuron_2],
                    layer.sublayer.neurons[i],
                    layer.neurons[i]
                    ]))
        chosen_neurons.add(branches) 

        # Add the error nodes
        error_nodes = VGroup()
        for n in chosen_neurons[-1]:
            error_nodes.add(n[-1].next)
        chosen_neurons.add(error_nodes)
        # Add error total node
        chosen_neurons.add(network.outputs[1].neurons[0])

        self.chosen_neurons = chosen_neurons

        anim_edges, fade_edges = get_edge_propogation_animations()
        self.play(anim_edges)
        self.play(fade_edges)
        self.play(anim_edges)
        self.play(fade_edges)

    def step_through(self, weight_index):
        '''
        step through traceback and derivatives in sequential order
        '''

        # A couple of helper functions to assist animations
        def highlight_graph(nodes):
            return  ApplyMethod(
                VGroup(*nodes).set_stroke, 
                YELLOW, 1.5*self.network_mob.edge_stroke_width,
                run_time = 0.5*self.network_mob.edge_propogation_time,
                )
        def wiggle_node(nodes):
            return ApplyMethod(
                VGroup(*nodes).scale_in_place, 
                1.2,
                rate_func=there_and_back,
                #run_time = self.network_mob.edge_propogation_time
                )

        self.chosen_neurons[0].save_state()
        self.chosen_neurons[2].save_state()
        self.chosen_neurons[3].save_state()
        for branch in self.chosen_neurons[1]:
            branch.save_state()
        self.network_mob.edge_groups.save_state()
        self.network_mob.out_edge_groups.save_state()

        neuron_1 = weight_index[2]-1              
        self.play(FadeIn(self.equals))
        
        # Indicate error total node
        self.play(Indicate(self.chosen_neurons[-1]), 
                rate_func=linear, 
                scale_factor=1,
                )
        # Indicate all intermediate layers
        i = 0 # Counts neurons with derivative terms associated
        j = 0 # Counts branches

        for branch, error in zip(self.chosen_neurons[1], self.chosen_neurons[2]):
            self.play(wiggle_node(error))
            self.play(highlight_graph([error[0].edges_out, error]))
            n0 = error[0].label.copy()
            for neuron in branch[::-1]:
                edge = j*(len(neuron[0].edges_out)-1)
                self.play(wiggle_node(neuron))
                if len(neuron[0].edges_out)!=0:
                     self.play(highlight_graph(neuron[0].edges_out[edge]))
                self.play(highlight_graph(neuron))
                self.play(ReplacementTransform(VGroup(*[n0,neuron.label.copy()]), self.chain_rule[i]))
                n0 = neuron.label.copy()
                i += 1
            j+=1
            if i > 1 and j < len(self.chosen_neurons[1]):
                self.play(FadeIn(self.plus))
        if i > 1:
            self.play(FadeIn(self.bra), FadeIn(self.ket))
        # Indicate initial layer
        self.play(wiggle_node(self.chosen_neurons[0]))
        self.play(highlight_graph(self.chosen_neurons[0]))
        
        # First layers play back      
        weight_edge = self.chosen_neurons[1][0][0].edges_in[neuron_1]
        self.play(ReplacementTransform(VGroup(*[self.chosen_neurons[1][0][0].label.copy(), 
                                self.chosen_neurons[0].label.copy()]), 
                                self.end_derivative[0]))
        self.play(highlight_graph(weight_edge))
        self.play(ReplacementTransform(VGroup(*[self.chosen_neurons[0].label.copy(),
                                    weight_edge.label.copy()]), 
                                    self.end_derivative[1]))
        l = [0, 2, 3]
        self.play(
        *[self.chosen_neurons[i].restore for i in l],
        *[branch.restore for branch in self.chosen_neurons[1]],
        self.network_mob.edge_groups.restore,
        self.network_mob.out_edge_groups.restore)

    def make_room(self, direction, scale=0.7):
        # direction will typically go '',RIGHT, DOWN, LEFT
        network_mob = self.network_mob
        equation = VGroup(*[self.de_dw, self.bp, self.end_derivative]) 
        equation.generate_target()
        equation.target.scale_in_place(scale)
        
        try:
            equation.target.next_to(self.equation[-1], direction, aligned_edge=LEFT)
            self.de_dw_terms.add(self.de_dw)#, self.bp[0])
            self.equals_terms.add(self.equals)
            self.de_da_terms.add(self.de_da_term)
            self.da_dz_terms.add(self.end_derivative[0])
            self.dz_dw_terms.add(self.end_derivative[1])
        except:
            self.equation = VGroup()
            equation.target.to_edge(direction)
            self.de_dw_terms = VGroup()
            self.de_dw_terms=self.de_dw#, self.bp[0])
            self.equals_terms=self.equals
            self.de_da_terms=self.de_da_term
            self.da_dz_terms=self.end_derivative[0]
            self.dz_dw_terms=self.end_derivative[1]

        self.play(MoveToTarget(equation))
        self.equation.add(equation)

    def organize_terms(self):

        try:
            self.de_dw_terms.add(self.de_dw)
            self.equals_terms.add(self.equals)
            self.de_da_terms.add(self.bp[1])
            self.da_dz_terms.add(self.end_derivative[0])
            self.dz_dw_terms.add(self.end_derivative[1])
        except:
            self.de_dw_terms = VGroup()
            self.de_dw_terms = self.de_dw
            self.equals_terms = self.equals
            self.de_da_terms = self.bp[1]
            self.da_dz_terms = self.end_derivative[0]
            self.dz_dw_terms = self.end_derivative[1]

    def get_brackets(self, mob, color):
        lb, rb = both = TexMobject("\\big[","\\big]", color=color)
        both.set_width(0.5)
        both.stretch_to_fit_height(1.2*mob.get_height())
        lb.next_to(mob, LEFT, 0.2*SMALL_BUFF)
        rb.next_to(mob, RIGHT, 0.2*SMALL_BUFF)
        return both

    def show_col_matrix(self, terms, color, 
                matrix=False, position=RIGHT):
        
        rects = VGroup()

        column_1 = VGroup(*[terms[i] for i in [0, 3]])
        column_2 = VGroup(*[terms[i] for i in [1, 2]])
        rect_1 = SurroundingRectangle(column_1, color=color, buff=0.5*SMALL_BUFF)
        rect_2 = SurroundingRectangle(column_2, color=color, buff=0.5*SMALL_BUFF)
        rects.add(VGroup(*[rect_1, rect_2]))
        
        self.play(ShowCreation(rect_1), ShowCreation(rect_2))
        
        if not matrix:
            column_move = column_1 
            column_move.generate_target()
            column_vec = column_move.target
        else:
            column_move = VGroup(*[column_1, column_2])
            column_move.generate_target()
            column_vec = column_move.target
            column_vec.arrange(RIGHT)
        
        try:
            if position is not LEFT:
                column_vec.next_to(self.columns[-1], 4*position)
            else:
                column_vec.next_to(self.matrix_equals, 2*position)
        except:
            self.columns = VGroup()
            self.column_brackets = VGroup()
            self.matrix_eqn = VGroup()
            self.rects = VGroup()
            column_vec.next_to(self.equation, 3*position)

        brackets = self.get_brackets(column_vec ,color=color)
        
        self.columns.add(column_vec)
        self.column_brackets.add(brackets)
        self.rects.add(rects)
        return [ReplacementTransform(VGroup(*[rect_1.copy(), rect_2.copy()]), brackets),
                 ReplacementTransform(column_move, column_vec)]

    def add_multipliers(self):
        hadamard = TexMobject("\\odot")
        multipliers = VGroup(*[hadamard.copy(), hadamard.copy()])
        #mult_arrows= Arrows(multipliers)
        self.move_to_center(self.column_brackets[0], multipliers[0], self.column_brackets[1])
        self.move_to_center(self.column_brackets[1], multipliers[1], self.column_brackets[2])
        self.play(FadeIn(multipliers))
        self.hadamard = multipliers

    def move_to_center(self, left_mobject, mid_mobject, right_mobject):
        mid = (right_mobject.get_left() - left_mobject.get_right())/2
        mid_mobject.move_to(left_mobject.get_right() + mid)

    def write_out_formulas(self):
        de_dw_mat = TexMobject("{\\partial E", "\\over", "\\partial W^{(2)}}")
        de_da_vec = TexMobject("{\\partial E", "\\over", "\\partial A^{(3)}}")
        da_dz_vec = TexMobject("{\\partial A", "\\over", "\\partial Z^{(3)}}")
        dz_dw_mat = TexMobject("{\\partial Z", "\\over", "\\partial W^{(2)}}")
        vec_derivatives = VGroup(*[
            de_da_vec,
            da_dz_vec,
            dz_dw_mat,
            de_dw_mat])
        vec_derivatives.scale(self.network_scale)

        self.matrix_equals = TexMobject("=")

        self.play(FadeOut(VGroup(*[
            self.equals_terms,
            self.de_da_terms,
            self.rects[0],
            self.da_dz_terms,
            self.rects[1],
            self.dz_dw_terms,
            self.rects[2]
            ]),
            lag_ratio=0.4))

        self.matrix_equals.next_to(self.columns[0], 2*LEFT)
        self.play(
            *self.show_col_matrix(self.de_dw_terms, GREEN_B, matrix=True, position = LEFT),
            FadeOut(self.rects[3]))
        self.play(FadeIn(self.matrix_equals))
        full_mat_eqn = VGroup(*[
            self.de_dw_terms, 
            self.matrix_equals, 
            self.columns, 
            self.column_brackets, 
            self.hadamard
        ])
        self.play(full_mat_eqn.align_to, self.network_mob, LEFT)
        self.play(FadeOut(self.full_network))
        arrows = VGroup()
        for i, derivative in enumerate(vec_derivatives):
            derivative.next_to(self.columns[i], 3*DOWN)
            arrow = Arrow(
                self.column_brackets[i].get_bottom(),
                derivative.get_top(),
                color=BLUE,
                buff = 0.5*SMALL_BUFF)
            arrows.add(arrow)

        self.play(FadeIn(vec_derivatives, lag_ratio=0.6),
                LaggedStartMap(GrowArrow, arrows, run_time=1))

        brace = Brace(VGroup(*vec_derivatives[0:2]), DOWN)
        delta = TexMobject("\\delta^L")
        delta.next_to(brace, DOWN)
        self.play(GrowFromCenter(brace), FadeIn(delta))

        final_eqn = VGroup(*[
            vec_derivatives[-1].copy(), 
            self.matrix_equals.copy(), 
            delta.copy(), 
            self.hadamard[0].copy(), 
            vec_derivatives[2].copy() 
            ])
        final_eqn.arrange(RIGHT)
        final_eqn.next_to(delta, DOWN, buff=LARGE_BUFF)
        self.play(FadeIn(final_eqn))

        eqn_rect = SurroundingRectangle(final_eqn, color=BLUE, buff=SMALL_BUFF)
        
        self.play(ShowCreation(eqn_rect))
        #TextMobject("Final Layer Formula Complete...")
        self.vec_derivatives = vec_derivatives
        self.eqn_rect = eqn_rect
        self.final_eqn = final_eqn
        self.full_mat_eqn = full_mat_eqn
        self.formula_delta = delta
        self.arrows = arrows
        self.brace = brace

    def reset_network(self):
        eqns = VGroup(*[
            self.full_mat_eqn,
            self.vec_derivatives,
            self.eqn_rect,
            self.formula_delta,
            self.final_eqn,
            self.arrows, 
            self.brace
        ])

        self.play(FadeOut(eqns))
        self.play(FadeIn(self.full_network))

    def recall_delta(self):
        # ...this needs refactoring.

        recall_eqn = self.full_mat_eqn
        equation = VGroup(*[self.de_dw, self.bp, self.end_derivative]) 
        equation.generate_target()
        equation.target.scale_in_place(self.network_scale)
        equation.target.shift(0.7*DOWN)
        
        self.full_network.save_state()
        self.play(self.full_network.scale_in_place, 0.8)
        self.play(self.full_network.fade, )
        equation.target.align_to(self.full_network, LEFT)
        recall_eqn.next_to(equation.target, UP)#aligned_edge=LEFT)
        self.play(MoveToTarget(equation))
        
        recall_text = TextMobject("Recall from \\\\ the last layer...")
        recall_text.next_to(recall_eqn, LEFT)
        self.play(Write(recall_text))
        self.play(FadeIn(recall_eqn))

        eqn_rects = VGroup()
        eqn_rects.add(SurroundingRectangle(
            self.columns[0][0], 
            color=BLUE,
            stroke_opacity=0.6 
            #buff=SMALL_BUFF
            ))
        eqn_rects.add(SurroundingRectangle(
            self.columns[1][0], 
            color=BLUE,
            stroke_opacity=0.6 
            #buff=SMALL_BUFF
            ))
        eqn_rects.add(SurroundingRectangle(
            self.bp[2][:2], 
            color=BLUE,
            stroke_opacity=0.6 
            #buff=SMALL_BUFF
            ))
        eqn_rects.add(SurroundingRectangle(
            self.columns[0][1], 
            color=GREEN,
            stroke_opacity=0.6 
            #buff=SMALL_BUFF
            ))
        eqn_rects.add(SurroundingRectangle(
            self.columns[1][1], 
            color=GREEN,
            stroke_opacity=0.6 
            #buff=SMALL_BUFF
            ))
        eqn_rects.add(SurroundingRectangle(
            self.bp[4][:2], 
            color=GREEN,
            stroke_opacity=0.6 
            #buff=SMALL_BUFF
            ))

        self.play(ShowCreation(VGroup(eqn_rects[0:2])))
        self.play(ShowCreation(eqn_rects[2]))
        self.play(ShowCreation(VGroup(eqn_rects[3:-1])))
        self.play(ShowCreation(eqn_rects[-1]))

        delta_1 = TexMobject("\\delta^L_1", color=BLUE)
        delta_2 = TexMobject("\\delta^L_2", color=GREEN)
        delta_column = VGroup(delta_1, delta_2)
        delta_column.scale(self.network_scale)
        delta_column.arrange(DOWN)
        delta_column.move_to(self.hadamard[0])
        delta_brackets = self.get_brackets(delta_column, WHITE)

        self.play(FadeOut(self.hadamard[0]),
            FadeOut(self.column_brackets[0:2]))
        self.play(FadeOut(VGroup(eqn_rects[0:2],eqn_rects[3:-1])))
        self.play(Transform(
                VGroup(self.columns[0][0], self.columns[1][0]), 
                VGroup(delta_column[0])
            ))
        self.play(Transform(
                VGroup(self.columns[0][1], self.columns[1][1]), 
                VGroup(delta_column[1])),
            FadeIn(delta_brackets),
            VGroup(self.columns[2], self.hadamard[1], self.column_brackets[2]).next_to,
                delta_brackets, RIGHT, 0.1) 

        self.play(FadeOut(eqn_rects[-1]))
        self.play(FadeOut(eqn_rects[2]))

        self.delta_transform(scale=self.network_scale)
        self.add(self.bp[2][:2])
        self.add(self.bp[4][:2])

        self.play(FadeOut(VGroup(
            self.columns,
            self.de_dw_terms,
            self.column_brackets[2:],
            self.hadamard[1], 
            delta_brackets, 
            self.matrix_equals,
            )), 
            FadeOut(recall_text))

        self.make_room(UP, scale=1)
        # equation = VGroup(*[
        #     self.de_dw, 
        #     self.bp, 
        #     self.end_derivative])
        # equation.generate_target()
        # equation.target.next_to(self.equation[-1], UP, aligned_edge=LEFT)
       
        # self.play(MoveToTarget(equation))
        # self.equation.add(equation)

        self.play(self.full_network.restore)        

    def delta_transform(self, scale=1):
        delta_1 = TexMobject("\\delta^L_1", color=BLUE)
        delta_2 = TexMobject("\\delta^L_2", color=GREEN)
        delta_1.move_to(self.bp[2]) 
        delta_2.move_to(self.bp[4])
        self.play(ReplacementTransform(self.bp[2][:2], delta_1))
        self.play(ReplacementTransform(self.bp[4][:2], delta_2))

        delta_2.generate_target()
        delta_2.target.scale_in_place(scale)
        delta_2.target.next_to(self.bp[3], RIGHT, 0.1*scale)

        self.play(MoveToTarget(delta_2))
        right_part_eqn = VGroup(*[self.bp[4][2:], self.bp[5:], self.end_derivative])

        self.play(right_part_eqn.next_to, delta_2, RIGHT, 0.1*scale)
        left_part_eqn = VGroup(*[self.bp[2][-1], self.bp[3], delta_2])

        delta_1.generate_target()
        delta_1.target.scale_in_place(scale)
        delta_1.target.next_to(self.bp[1], RIGHT, 0.1*scale)
        self.play(MoveToTarget(delta_1))
        self.play(VGroup(left_part_eqn, right_part_eqn).next_to, delta_1, RIGHT, 0.1*scale)
        
        self.bp[2][:2].replace(delta_1)
        self.bp[4][:2].replace(delta_2)

        self.remove(delta_2)
        self.remove(delta_1)

    def partial_derivative(self, weight_index):
        '''
        weight index is a list [layer, j, k]
        negative layer indexs work
        '''   

        layer_index = weight_index[0]
        neuron_2 = weight_index[1]-1
        neuron_1 = weight_index[2]-1

        layer = self.network_mob.layers[layer_index]
        n_neurons = len(layer.neurons)
        L_terms = VGroup() # Penultimate layer terms
        error = self.e_labels.copy()
        bra = TexMobject("\\big(")
        ket = TexMobject("\\big)")
        plus = TexMobject("+")
        equals = TexMobject("=")
        # Add error and layer/sublayer from last layer for each branch 
        # [(error1, n_1, n_s1), (error2, n_2, n_s2)]
        for n in range(len(self.network_mob.layers[-1].neurons)):
            L_terms.add(VGroup(*[
                error[n],
                self.neuron_labels[-1][n].copy(),
                self.sublayer_labels[-1][n].copy()
            ]))

        weight = TexMobject("w^{(%d)}_{%d%d}" 
                %(layer_index, neuron_2+1, neuron_1+1))

        # If we're only looking at the last layer only concerned
        # with neuron_2 error node path
        if (layer_index+1) == len(self.network_mob.layers):
            L_terms = VGroup(L_terms[neuron_2][:2])
            chain_rule=VGroup() 
        
        delta_L = VGroup() 
        for branch in L_terms:
            chain_derivative_L = VGroup() 
            for term1,term2 in zip(branch[:-1], branch[1:]):
                chain_derivative_L.add(
                     TexMobject("{\\partial",term1.get_tex_string(), 
                                 "\\over", 
                                 "\\partial", term2.get_tex_string(), "}")
                )
                chain_derivative_L.arrange(RIGHT)            
            delta_L.add(chain_derivative_L)
        delta_L.scale(self.label_scale)
        # These backprop (bp) and chain_rule terms are only needed     
        # if there is one layer. If there is more than 1 layer to  
        # backpropogate these terms will be overwritten below
        bp = VGroup(*[equals, delta_L])
        self.de_da_term = bp[1] 
        chain_rule = delta_L        
        if (layer_index+1) < len(self.network_mob.layers):
            l_terms = VGroup() # intermediate layer terms
            # Add all labels up to layer index layer
            # only applicable if network greater than 2 layers
            for i, layer in enumerate(self.network_mob.layers[layer_index+1:-1],
                                    start=1):
                for j in range(len(layer.neurons)):
                    l_terms.add(
                        self.neuron_labels[layer_index+i][j].copy(),
                        self.sublayer_labels[layer_index-1+i][j].copy()
                    )

            # labels right before the first layer (layer_index layer)
            delta_l = VGroup()
            for n in range(len(self.sublayer_labels[layer_index])):    
                l_terms.add(
                    self.sublayer_labels[layer_index][n].copy(),
                )
            l_terms.add(self.neuron_labels[layer_index][neuron_2].copy())

            for term1,term2 in zip(l_terms[:-1], it.cycle(l_terms[-1])):
                delta_l.add(VGroup(*[
                    TexMobject("{\\partial",term1.get_tex_string(), 
                                "\\over", 
                                "\\partial", term2.get_tex_string(), "}")
                ]))
            delta_l.scale(self.label_scale)
            
            # Magic to combine ...
            chain_rule = []
            chain_rule = it.chain.from_iterable(list(zip(delta_L, delta_l)))
            # This part flattens the delta_L sublist so each term can be 
            # displayed individually later.
            chain_rule = VGroup(*it.chain.from_iterable(chain_rule))
            chain_rule.arrange(RIGHT)
            # Combine all back prop terms
            bp = VGroup(*[
                    equals,
                    bra,
                    chain_rule[:3],
                    plus,
                    chain_rule[3:],
                    ket,
            ])
            self.de_da_term = bp[2][:2] 

        # Add the labels in the layer = layer_index (i.e. end of chain)
        chain_end = VGroup()            
        chain_end.add(
            self.neuron_labels[layer_index][neuron_2].copy(),
            self.sublayer_labels[layer_index-1][neuron_2].copy(),
        )
        chain_end.add(weight)
        end_derivative = VGroup()
        for term1,term2 in zip(chain_end[:-1], chain_end[1:]):
            end_derivative.add(
                VGroup(*[TexMobject("{\\partial", term1.get_tex_string(), 
                            "\\over", 
                            "\\partial", term2.get_tex_string(), "}")
            ]))
        eqn_buff = 0.2
        end_derivative.scale(self.label_scale)
        end_derivative.arrange(RIGHT, buff=eqn_buff)  
        bp.arrange(RIGHT, buff=eqn_buff)
        bp.next_to(self.de_dw, buff=eqn_buff)
        end_derivative.next_to(bp, buff=eqn_buff)
        self.eqn_buff = eqn_buff
        self.bra = bra
        self.ket = ket
        self.plus = plus
        self.equals = equals
        self.bp = bp
        self.chain_rule = chain_rule
        self.end_derivative = end_derivative

    def activate_bias(self, layer_index):

        layer = self.network_mob.bias[layer_index-1]
        #activations = 0.7*np.random.random(len(layer.neurons))
        #active_layer = self.network_mob.get_active_layer(layer_index, activations)

        b_labels = TexMobject("b^{(%d)}"%(layer_index))

        for label, neuron in zip(b_labels, layer.neurons):
            label.scale(self.label_scale)
            label.move_to(neuron)

        #self.play(
            #Transform(layer, active_layer),
        #    Write(b_labels, run_time = 0.5)
        #)
        self.add(b_labels) 
        try:
            self.b_labels.add(b_labels)
        except:
            self.b_labels = VGroup()
            self.b_labels.add(b_labels)

    def bce(self):
        ''' 
        Binary Cross Entropy error function label
        '''

        # Define existing labels we will be copying and moving around
        neuron_labels = VGroup(*[
            self.neuron_labels[-1][0], 
            self.neuron_labels[-1][0]
        ]).copy()
        neuron_labels.generate_target()
        error_sum1 = VGroup()
        error_sum2 = VGroup()

        error_type = TextMobject("Binary Cross-Entropy")
        neuron_label = neuron_labels.target
        neuron_label.scale(1./0.75) 
        y_label = TexMobject("-","y")
        y_label2 = TexMobject("(1-", "y",")")
        minus =  TexMobject("-")
        nat_log = TexMobject("ln")
        bra = TexMobject("(")
        ket = TexMobject(")")
        error_sum1.add(y_label, nat_log, bra, neuron_label[0], ket, minus)
        error_sum2.add(y_label2, nat_log.copy(), bra.copy(),
                    TexMobject("1-"), neuron_label[1], ket.copy())
        
        error_sum1.arrange(RIGHT, buff=SMALL_BUFF)
        error_sum2.arrange(RIGHT, buff=SMALL_BUFF)
        # Combine the error_sums so that we can manipulate easier
        error_sum = VGroup(*[error_sum1, error_sum2])
        error_sum.arrange(DOWN)
        error_sum[0].align_to(error_sum[1], LEFT)

        y_label[-1].set_color(YELLOW)
        y_label2[-2].set_color(YELLOW)
        #neuron_label.set_color(BLUE)

        return error_type, error_sum, neuron_labels

    def mse(self):
        ''' 
        Mean Squared Error function label
        '''

        # Define existing labels we will be copying and moving around
        neuron_labels = self.neuron_labels[-1][-1].copy()
        neuron_labels.generate_target()
        error_sum = VGroup()

        error_type = TextMobject("Mean Squared Error")
        neuron_label = neuron_labels.target
        neuron_label.scale(1./0.75) 
        half = TexMobject("\\frac{1}{2}")
        y_label = TexMobject("y")
        minus =  TexMobject("-")
        bra = TexMobject("(")
        ket = TexMobject(")")
        square = TexMobject(")^2")
        error_sum.add(half, bra, y_label, minus, neuron_label, square)
        #error_sum = VGroup(
        #    half, bra, y_label, minus, neuron_label, square
        #    )
        error_sum.arrange(RIGHT, buff=SMALL_BUFF)

        y_label.set_color(YELLOW)
        #neuron_label.set_color(BLUE)

        return error_type, error_sum, neuron_labels

    def show_error(self, neuron_index, position, error_func):
        '''
        neuron_index = the output neuron that we will use 
        for positioning equations.
        position = where to display the equation relative to 
        the neuron.
        error_func is a string that evaluates to the error function
        i.e. 'mse' or 'bce'
        '''
        error_func = eval("self." + error_func + "()")
        error_type, error_sum, neuron_labels = error_func
        neuron = self.network_mob.outputs[0].neurons[neuron_index]
        e_labels = self.e_labels[neuron_index].copy()
        # Copies e_labels and returns in e_labels.target.  The targets are
        # manipulated to the form desired then MoveToTarget can be used to 
        # transform the objects to the new location.
        e_labels.generate_target()  
        e_label = e_labels.target
        e_label.scale(1./0.75)
        
        scale_factor = self.label_scale # Used to scale the equations text
        error_sum.scale(scale_factor)
        e_label.scale(scale_factor)

        # Position everything
        e_label.next_to(neuron, RIGHT)
        equals = TexMobject("=")
        equals.next_to(e_label, RIGHT, buff=SMALL_BUFF)
        equals.scale(scale_factor)
        error_sum.next_to(equals, RIGHT)

        #error_sum.to_edge(RIGHT)        
        error_type.next_to(neuron, position)

        e_label.set_color(ORANGE)
        neuron_labels.target.shift(0.5*SMALL_BUFF*UP)

        self.play(
            Write(error_type),
            neuron.set_fill, None, 0.1,
        )
        self.wait()
        # Use ReplacementTransform to replace the mobject
        # with the target (i.e. don't end up with two copies
        # of the mobject)
        self.play(
            ReplacementTransform(error_type,equals),
            MoveToTarget(e_labels, run_time = 1.5),
            #neuron.set_fill, None, 0.3,
            run_time = 1
        )
        self.play(
            MoveToTarget(neuron_labels, run_time = 1.5),
            #ReplacementTransform(neuron_labels, neuron_labels.target, run_time=1.5)
        )

        self.play(
            Write(error_sum),
        )
        
        # There are two copies of neuron_labels, the target copy in error_sum 
        # and the moved copy after MoveToTarget.  Get rid of the copy 
        # not part of the error_sum group. ReplacementTransform doesn't
        # work with this particular setup
        self.remove(neuron_labels)

        error_sum.add_to_back(equals, e_labels)
        try:
            self.error_sum.add(error_sum)
        except:
            # This has to be assigned as a VGroup otherwise 
            # error_sum is treated as a list of components 
            # i.e. it unpacks the original error_sum VGroup
            self.error_sum = VGroup(*[error_sum])

        # self.a1_label = a1_label
        # self.a1_equals = equals
        # self.w_labels = w_labels
        # self.a_labels_in_sum = neuron_labels
        # self.symbols = symbols
        # self.weighted_sum = VGroup(w_labels, neuron_labels, symbols)

    
    def show_first_neuron_weighted_sum(self):
        neuron = self.network_mob.layers[1].neurons[0]
        neuron_labels = VGroup(*self.neuron_labels[:2]).copy()
        # Copies neuron_labels and returns in neuron_labels.target.  The targets are
        # maniputated to the form desired then MoveToTarget can be used to 
        # transform the objects to the new location.
        neuron_labels.generate_target() 
        w_labels = VGroup(*[
            TexMobject("w_{0, %d}"%d)
            for d in range(len(neuron_labels))
        ])
        weighted_sum = VGroup()
        symbols = VGroup()
        for neuron_label, w_label in zip(neuron_labels.target, w_labels):
            neuron_label.scale(1./0.75)
            plus =  TexMobject("+")
            weighted_sum.add(w_label, neuron_label, plus)
            symbols.add(plus)
        weighted_sum.add(
            TexMobject("\\cdots"),
            TexMobject("+"),
            TexMobject("w_{0, n}"),
            TexMobject("a^{(0)}_n"),
        )

        weighted_sum.arrange(RIGHT)
        a1_label = TexMobject("a^{(1)}_0")
        a1_label.next_to(neuron, RIGHT)
        equals = TexMobject("=").next_to(a1_label, RIGHT)
        weighted_sum.next_to(equals, RIGHT)

        symbols.add(*weighted_sum[-4:-2])
        w_labels.add(weighted_sum[-2])
        neuron_labels.add(self.neuron_labels[-1].copy())
        neuron_labels.target.add(weighted_sum[-1])
        neuron_labels.add(VGroup(*self.neuron_labels[2:-1]).copy())
        # I think VectorizedPoint creates an array of points based on the 
        # list/array sent to it. Here its just adding the cdots to the 
        # label target list
        neuron_labels.target.add(VectorizedPoint(weighted_sum[-4].get_center()))

        VGroup(a1_label, equals, weighted_sum).scale(
            0.75, about_point = a1_label.get_left()
        )

        w_labels.set_color(GREEN)
        w_labels.shift(0.6*SMALL_BUFF*DOWN)
        neuron_labels.target.shift(0.5*SMALL_BUFF*UP)

        self.play(
            Write(a1_label), 
            Write(equals),
            neuron.set_fill, None, 0.3,
            run_time = 1
        )
        self.play(MoveToTarget(neuron_labels, run_time = 1.5))
        self.play(
            Write(w_labels),
            Write(symbols),
        )

        self.a1_label = a1_label
        self.a1_equals = equals
        self.w_labels = w_labels
        self.a_labels_in_sum = neuron_labels
        self.symbols = symbols
        self.weighted_sum = VGroup(w_labels, neuron_labels, symbols)

    def add_bias(self):
        weighted_sum = self.weighted_sum
        bias = TexMobject("+\\,", "b_0")
        bias.scale(0.75)
        bias.next_to(weighted_sum, RIGHT, SMALL_BUFF)
        bias.shift(0.5*SMALL_BUFF*DOWN)
        name = TextMobject("Bias")
        name.scale(0.75)
        name.next_to(bias, DOWN, MED_LARGE_BUFF)
        arrow = Arrow(name, bias, buff = SMALL_BUFF)
        VGroup(name, arrow, bias).set_color(BLUE)

        self.play(
            FadeIn(name),
            FadeIn(bias),
            GrowArrow(arrow),
        )

        self.weighted_sum.add(bias)

        self.bias = bias
        self.bias_name = VGroup(name, arrow)

    def add_sigmoid(self):
        weighted_sum = self.weighted_sum
        weighted_sum.generate_target()
        #sigma, lp, rp = mob = TexMobject("\\sigma\\big(\\big)")
        
        sigma, lp, rp = mob = TexMobject("\\sigma","\\big(", "\\big)")
        # mob.scale(0.75)
        sigma.move_to(weighted_sum.get_left())
        sigma.shift(0.5*SMALL_BUFF*(DOWN+RIGHT))
        lp.next_to(sigma, RIGHT, SMALL_BUFF)
        weighted_sum.target.next_to(lp, RIGHT, SMALL_BUFF)
        rp.next_to(weighted_sum.target, RIGHT, SMALL_BUFF)

        name = TextMobject("Sigmoid")
        name.next_to(sigma, UP, MED_LARGE_BUFF)
        arrow = Arrow(name, sigma, buff = SMALL_BUFF)
        sigmoid_name = VGroup(name, arrow)
        VGroup(sigmoid_name, mob).set_color(YELLOW)

        self.play(
            FadeIn(mob),
            MoveToTarget(weighted_sum),
            MaintainPositionRelativeTo(self.bias_name, self.bias),
        )
        self.play(FadeIn(sigmoid_name))

        self.sigma = sigma
        self.sigma_parens = VGroup(lp, rp)
        self.sigmoid_name = sigmoid_name

    def show_meaning_of_matrix_row(self):
        #row = self.top_matrix_row
        edges = self.network_mob.layers[1].neurons[0].edges_in.copy()
        edges.set_stroke(GREEN, 5)
        #rect = SurroundingRectangle(row, color = GREEN_B)

        #self.play(ShowCreation(rect))
        for x in range(2):
            self.play(LaggedStartMap(
                ShowCreationThenDestruction, edges,
                lag_ratio = 0.8
            ))
        self.wait()

    def activate_neuron_weights(self, layer_index):
        layer = self.network_mob.layers[layer_index]
        size_l = self.network_mob.layer_sizes[layer_index]
        size_lp = self.network_mob.layer_sizes[layer_index-1]
        edges = self.network_mob.edge_groups[layer_index-1]
        anim_edges = self.network_mob.get_edge_propogation_animations(layer_index-1)
        eps = 1e-3

        w_labels = VGroup(*[
            TexMobject("w^{(%d)}_{%d%d}" %(layer_index, l2, l1))
            for l1, l2 in it.product(range(1, size_l+1), range(1, size_lp+1))
        ])

        # It's easier to create a list of just the weight labels and 
        # a list of the labels after they've been rotated for the neural
        # network
        try:
            self.w_labels.add(w_labels) 
        except:
            self.w_labels = VGroup()
            self.w_labels.add(w_labels)

        w_labels_nn = w_labels.copy()
        # Position w_labels
        for w_label, edge in zip(w_labels_nn, edges):
            w_label.scale(self.label_scale)
            w_label.rotate(edge.get_angle())
            w_label.vector = edge.get_unit_vector()
            w_label.perp = rotate_vector(w_label.vector, np.pi/2)
            
            if abs(edge.get_angle()) > eps:
                w_label.shift(edge.get_center() + 
                    1*(w_label.vector) + 
                    0.4*(w_label.perp))
            else:
                w_label.shift(edge.get_center() + UP*0.4)

            w_label.set_color(GREEN)
            edge.label = w_label
        # This is the neural network weight list
        try:
            self.w_labels_nn.add(w_labels_nn) 
        except:
            self.w_labels_nn = VGroup()
            self.w_labels_nn.add(w_labels_nn)

        #self.play(*anim_edges)
        self.play(*anim_edges,
            Write(w_labels_nn, run_time = 1)
        )

    def activate_bias_weights(self, layer_index):
        layer = self.network_mob.layers[layer_index]
        size_l = self.network_mob.layer_sizes[layer_index]
        # layer_index-1 = current bias index.
        edges = self.network_mob.bias_edge_groups[layer_index-1]
        anim_edges = self.network_mob.get_edge_propogation_animations(layer_index-1, bias=True)
        eps = 1e-3

        bw_labels = VGroup(*[
            TexMobject("b^{(%d)}_{%d}" %(layer_index, l1))
            for l1 in range(1, size_l+1)
        ])

        for bw_label, edge in zip(bw_labels, edges):
            bw_label.scale(self.label_scale)
            bw_label.rotate(edge.get_angle())
            bw_label.vector = edge.get_unit_vector()
            bw_label.perp = rotate_vector(bw_label.vector, np.pi/2)
            
            if abs(edge.get_angle()) > eps:
                bw_label.shift(edge.get_corner(DL) + 
                    0.5*(bw_label.vector) +
                    0.2*(bw_label.perp))
                    
            else:
                bw_label.shift(edge.get_center() + UP*0.4)

            bw_label.set_color(BLUE)

        self.play(*anim_edges,
            Write(bw_labels, run_time = 1)
        )

        try:
            self.bw_labels.add(bw_labels)
        except:
            self.bw_labels = VGroup()
            self.bw_labels.add(bw_labels)

    def show_weights(self):
        self.chosen_neurons = VGroup()
        layer = self.network_mob.layers[1]
        for neuron in layer.neurons:
            self.chosen_neurons.add(neuron)

        for neuron in self.chosen_neurons:
            self.play(Indicate(neuron))
        self.activate_neuron_weights(1)
        self.activate_neuron_weights(2)
        self.activate_bias_weights(1)
        self.activate_bias_weights(2)
        self.wait()
        #self.play(Indicate(subscripts))
        #for x in range(2):
        #    self.play(Swap(*subscripts))
        #    self.wait()

        #self.set_variables_as_attrs(faded_edges, w_label)