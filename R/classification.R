

#' Title  make classification for triplet-cGAN
#'
#' @param node  list of nodes in each hidden layer
#' @param input_shape  Integer.Number of genes in modeling
#'
#' @return classification network
#' @export
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#' tfa <- import('tensorflow_addons')  #   import tensorflow_addons model
#' discriminator <- SNGAN_discriminator(node=c(256,128,64,16,8),
#'                                      input_shape= 15400L,dropout_rate = 0.2)
#' discriminator
#' }

classification <- function(node=c(512,256,128,64,10),input_shape= 15400L ){

  shape <- input_shape
  input_class <- keras::layer_input(shape = shape)


  for (i in 1:length(node)) {
    if(i ==1){
      class <- input_class%>%
        keras::layer_dense(units = node[i],input_shape = shape ) %>%
        keras::layer_activation_relu( )
    }
    if(i >1){
      class <- class%>%
        keras::layer_dense(units  = node[i] ) %>%
        keras::layer_activation_relu( )
    }

  }

  class <- class%>%
    keras::k_l2_normalize()


  classification <- keras::keras_model(input_class,class)
  classification

  return(classification)
}

