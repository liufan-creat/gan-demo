


#' Title make generator with condition
#'
#' @param node list of nodes in each hidden layer
#' @param output_shape Number of cells in modeling
#' @param n.label Number of groups in RNA-seq
#'
#' @param scRNA  data for raw or normalized scRNA-seq data,
#' the ordering of all genes needs to be the same as bulk RNA-seq
#' @param custom_count Whether to customize the total number of each single cell (TURE by default)
#' if set to False, calculate total count of each single cell form normal_single_cell after gene filtering
#' @param count total count of each single cell before gene filtering
#'
#' @return conditionial generator network
#' @export
#'
#' @examples
#' \dontrun{
#' library(bulkGAN)
#'
#'generator <- cGAN_generator(node=c(128,256,512),output_shape= 800L,
#'                           scRNA=normal_single_cell,custom_count=TRUE,count=count,n.label =3)
#'generator
#'}
cGAN_generator <- function(node=c(128,256,512),output_shape= 800L,
                           scRNA=NULL,custom_count=TRUE,count=NULL,n.label =3){


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
    my_variable <-  keras::k_repeat(count,length(input[,1]))
    my_variable <-  keras::k_sum(my_variable,axis = -1)
    count <- 10000/my_variable

    return(count * my_sum)
  }

  l <- 100L

  input_g <- keras::layer_input(shape = l)
  label1 <- keras::layer_input(shape=list(1))

  label_embedding <- label1 %>%
    keras::layer_embedding(input_dim = n.label,output_dim =  l)  %>%
    keras::layer_flatten()

  model_input <- keras::layer_concatenate(list(input_g,label_embedding) )

  for (i in 1:length(node)) {
    if(i ==1){
      gen <- keras::keras_model_sequential()
      gen <- model_input %>%
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
    keras::layer_activation_relu( )%>%
    keras::layer_lambda( f = Tsln  )


  generator <- keras::keras_model(list(input_g,label1),gen)
  generator


  return(generator)
}
