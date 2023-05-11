

#' Title get cell weight with h5 file
#'
#' @param node list of nodes in each hidden layer
#' @param output_shape Number of cells in modeling
#' @param file.path File path of file by training for GAN
#'
#' @return generator network for generator cell weight
#' @export
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#
#'generator <- SNGAN_generator(node=c(128,256,512),output_shape= 800L,
#'                             file.path = "~/GAN/2023_2_16/getx/cosin/generator_cosin_1000.h5")
#'
#'generator
#'}

get_cell_weight <- function(node=c(128,256,512),output_shape= 800L,
                            file.path = NULL ){

  l <- 100L
  input <- keras::layer_input(shape = l)

  for (i in 1:length(node)) {
    if(i ==1){
      gen <- input%>%
        keras::layer_dense(units = node[1] ,input_shape = l) %>%
        keras::layer_activation_relu( )
    }
    if(i >1){
      gen <- gen%>%
        keras::layer_dense(units = node[i] ) %>%
        keras::layer_activation_relu( )
    }
  }

  gen <- gen%>%
    keras::layer_dense(units = output_shape) %>%
    keras::layer_activation_relu( )

  generator <- keras::keras_model(input,gen)
  generator

  keras::load_model_weights_hdf5(generator,file.path)

  return(generator)
}

