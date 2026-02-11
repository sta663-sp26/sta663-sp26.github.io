options(
  width=80
)

knitr::opts_chunk$set(
  fig.align = "center"
)

local({
  hook_err_old <- knitr::knit_hooks$get("error")  # save the old hook
  knitr::knit_hooks$set(error = function(x, options) {
    # now do whatever you want to do with x, and pass
    # the new x to the old hook
    x = sub("## \n## Detailed traceback:\n.*$", "", x)
    x = sub("Error in py_call_impl\\(.*?\\)\\: ", "", x)
    hook_err_old(x, options)
  })
  
  hook_warn_old <- knitr::knit_hooks$get("warning")  # save the old hook
  knitr::knit_hooks$set(warning = function(x, options) {
    x = sub("<string>:1: ", "", x)
    hook_warn_old(x, options)
  })
})
