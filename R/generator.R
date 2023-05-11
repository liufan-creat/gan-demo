


#' Title make generator
#'
#' @param node list of nodes in each hidden layer
#' @param output_shape Number of cells in modeling
#' @param scRNA  data for raw or normalized scRNA-seq data,
#' the ordering of all genes needs to be the same as bulk RNA-seq
#' @param custom_count Whether to customize the total number of each single cell (TURE by default)
#' if set to False, calculate total count of each single cell form normal_single_cell after gene filtering
#' @param count total count of each single cell before gene filtering
#'
#' @return generator network
#' @export
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#'
#'generator <- SNGAN_generator(node=c(128,256,512),output_shape= 800L,
#'                            scRNA=NULL,custom_count=TRUE,count=NULL )
#'generator
#'}
SNGAN_generator <- function( node=c(128,256,512) , output_shape= 800L,
                            scRNA=NULL,custom_count=TRUE,count=NULL ){

  if(custom_count == F){
    count <- apply(scRNA,2,sum)
  }

  input1 <- keras::as_tensor(   as.matrix(scRNA)  , dtype = 'float32')
  input1 <- keras::k_expand_dims(input1,axis = 1)

  input2 <- keras::as_tensor(   t(count)  , dtype = 'float32')

  Tsln <- function(output){
    my_variable <-  keras::k_repeat(output,length(scRNA[,1]))
    res <-  my_variable*input1 ## count_matrix
    my_sum <-   keras::k_sum(res,axis = -1)

    count <- output*input2  ## count sum
    my_variable <-  keras::k_repeat(count,length(scRNA[,1]))
    my_variable <-  keras::k_sum(my_variable,axis = -1)
    count <- 10000/my_variable

    return(count * my_sum)
  }

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

  gen <- gen %>%
    keras::layer_dense(units = output_shape) %>%
    keras::layer_activation_relu( ) %>%
    keras::layer_lambda( f = Tsln  )

  generator <- keras::keras_model(input,gen)
  generator

  return(generator)
}

