import {parse} from "query-string";
import { useMemo } from "react";
import {useLocation} from "react-router-dom";

const useQuery = () => {
  const {search} = useLocation();
  const qsObj = useMemo(() => parse(search), [search]);
  return qsObj;
}

export default useQuery;