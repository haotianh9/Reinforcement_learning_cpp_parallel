std::tuple<std::vector<double>, double> getAction(std::vector<double> observation, int dim)
{
    std::vector<double> action(dim);
    action[0]=observation[0]+observation[1];
    float logprob;
    logprob=0.2;
    return {action, logprob};
}