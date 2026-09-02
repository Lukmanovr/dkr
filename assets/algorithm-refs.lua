-- Cross-references to algorithm floats (`@alg-…`) come out of Quarto's crossref
-- filter with the fallback prefix "alg." (the algorithm environment is a custom
-- theorem-like div, not one of Quarto's built-in kinds). This runs after Quarto's
-- own filters and rewrites that prefix to "Algorithm", so a reference reads
-- "Algorithm 2" beside "Equation 7" and "Proposition 4". Nothing else is touched.
function Link(el)
  local target = el.target or ""
  if target:match("^#alg%-") then
    local out = pandoc.List()
    for _, inline in ipairs(el.content) do
      if inline.t == "Str" then
        inline.text = inline.text:gsub("^[Aa]lg%.", "Algorithm")
      end
      out:insert(inline)
    end
    el.content = out
  end
  return el
end
