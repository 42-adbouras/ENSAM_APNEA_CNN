import { createContext, useContext, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Model } from "./api";

interface ModelState {
  models: Model[];
  loading: boolean;
  error: string | null;
  chosen: string | null;
  choose: (name: string) => void;
}

const ModelContext = createContext<ModelState | null>(null);

export function ModelProvider({ children }: { children: ReactNode }) {
  const query = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [picked, setPicked] = useState<string | null>(null);

  const models = query.data?.models ?? [];
  const usable = models.filter(m => m.threshold !== null).map(m => m.name);
  const fallback = query.data && usable.includes(query.data.default) ? query.data.default : usable[0] ?? null;
  const chosen = picked !== null && usable.includes(picked) ? picked : fallback;
  // The model picked in the dropdown while it can still be used, otherwise the default model.

  return (
    <ModelContext.Provider value={{
      models,
      loading: query.isPending,
      error: query.error?.message ?? null,
      chosen,
      choose: setPicked,
    }}>
      {children}
    </ModelContext.Provider>
  );
}

export function useModel(): ModelState {
  const state = useContext(ModelContext);
  if (!state) throw new Error("useModel needs a ModelProvider");
  return state;
}
