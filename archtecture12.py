import graphviz


def draw_cnn_model(model):
    dot = graphviz.Digraph()

    # Create a unique ID for each layer
    layer_ids = {}
    for i, layer in enumerate(model.layers):
        layer_ids[layer.name] = str(i)

    # Create nodes for each layer
    for layer in model.layers:
        layer_id = layer_ids[layer.name]
        dot.node(layer_id, layer.name)

    # Connect the layers
    for layer in model.layers:
        layer_id = layer_ids[layer.name]

        if layer == model.layers[0]:
            # Connect input layer to the first hidden layer
            input_shape = model.input_shape[1:]  # Exclude batch dimension
            input_layer_id = layer_ids[model.layers[0].name]
            dot.edge(input_layer_id, layer_id, label=f'{input_shape}')
        else:
            # Connect the input layers to the current layer
            inbound_layers = layer._inbound_nodes[0].inbound_layers
            if isinstance(inbound_layers, list):  # Multiple input layers
                for input_layer in inbound_layers:
                    input_layer_id = layer_ids[input_layer.name.split('/')[0]]
                    dot.edge(input_layer_id, layer_id)
            elif isinstance(inbound_layers, dict):  # Single input layer
                input_layer = list(inbound_layers.values())[0]
                input_layer_id = layer_ids[input_layer.name.split('/')[0]]
                dot.edge(input_layer_id, layer_id)

    # Render and save the graph
    dot.format = 'png'
    dot.render('cnn_model', view=True)


# Example usage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

model = Sequential()
model.add(Input(shape=(32, 32, 3), name='input_layer'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

draw_cnn_model(model)
