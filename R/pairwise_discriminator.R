


#' Title  make discriminator for pairwise-GAN
#'
#' @param node  list of nodes in each hidden layer
#' @param input_shape  Integer.Number of genes in modeling
#' @param dropout_rate  float between 0 and 1. Fraction of the input units to drop.
#'
#' @return discriminator network
#' @export
#'
#' @examples
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#' tfa <- import('tensorflow_addons')  #   import tensorflow_addons model
#' discriminator <- SNGAN_discriminator(node=c(256,128,64,16,8),
#'                                      input_shape= 15400L,dropout_rate = 0.2)
#' discriminator
#'
#'
#'
SNGAN_discriminator <- function(node=c(256,128,64,16,8),
                                input_shape= 15400L,dropout_rate = 0.2 ){

  shape <- input_shape

  input_pos <- keras::layer_input(shape = shape)
  input_neg <- keras::layer_input(shape = shape)

  model_input2 <- keras::layer_subtract(list(input_pos,input_neg))


  for (i in 1:length(node)) {
    if(i ==1){
      dis <- keras::keras_model_sequential()
      dis <- model_input2%>%
        keras:: k_square()%>%
        keras::k_l2_normalize()%>%
        tfa$layers$SpectralNormalization(layer_dense(units = node[i],activation = 'relu' ))()%>%
        keras::layer_dropout(rate = dropout_rate)
    }

    if(i >1){
      dis <- dis%>%
      tfa$layers$SpectralNormalization(layer_dense(units = node[i],activation = 'relu' ))()%>%
      keras::layer_dropout(rate = dropout_rate)
      }


  }


  dis <- dis%>%
    keras::layer_flatten()%>%
    keras::layer_dense(units = 1 )


  discriminator <- keras::keras_model(list(input_pos,input_neg),dis)
  discriminator

  return(discriminator)
}

