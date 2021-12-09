
std::vector<double> getAction(std::vector<double> observation, int dim)
{
    std::vector<double> action(dim);
    action[0]=observation[0]+observation[1];
    return action;
}